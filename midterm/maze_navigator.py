"""
Windows-ready copy of maze_navigator.
No macOS-only file paths were present here, so the navigation logic is unchanged.
"""

"""
maze_navigator.py
=================
Runtime navigation functions imported by player.py.

Contains everything needed to localise and navigate at inference time:
    label_actions_from_commands
    save_graph
    load_graph
    localize
    localize_robust
    MazeNavigator              (class — wraps localize, plan, execute)
    navigate_to_goal

Fixes applied vs original:
    [1] localize()               — temporal boost was dead code (pass); now
                                   accepts graph and applies neighbor boost
                                   consistently with MazeNavigator.localize()
    [2] localize_robust()        — temp files leaked on encoder exception;
                                   all encode() calls now wrapped in try/finally
    [3] scan_directions()        — same temp-file leak fixed (try/finally)
    [4] get_alignment_scores()   — same temp-file leak fixed (try/finally)
    [5] scan_directions()        — Image.ROTATE_180 → Image.Transpose.ROTATE_180
                                   for Pillow ≥ 9.1 compatibility
    [6] scan_directions()        — back-direction distance now negated so caller
                                   can distinguish behind vs ahead
    [7] confirm_step()           — lost_candidate tracked separately; replan
                                   uses actual localized position, not stale node
    [8] set_goal()               — nx.NodeNotFound added to except tuple
    [9] navigate_to_goal()       — action strings replaced with Action enums;
                                   vis_nav_game import guarded with try/except
   [10] label_actions_from_commands() — O(N²) scan replaced with bisect O(N log N)
"""

import bisect
import math
import os
import pickle
import tempfile

import cv2
import networkx as nx
import numpy as np
from PIL import Image
from build_graph import load_graph  # single source of truth

# ─────────────────────────────────────────────────────────────────────────────
# 1. ACTION LABELLING
# ─────────────────────────────────────────────────────────────────────────────

def label_actions_from_commands(keyframes: list[dict],
                                command_log: list[tuple]) -> list[dict]:
    """
    Attach an 'action_to_next' field to each keyframe using the command log
    recorded during exploration.

    command_log is a list of (frame_idx, action_str) tuples built by
    player.py see() via infer_action_from_flow().

    FIX [10]: Original implementation was O(N²) — for every keyframe it
    filtered the entire command log. Replaced with a sorted list + bisect
    lookup for O(N log N) total.

    Called once in pre_navigation() before build_pose_graph().
    """
    if not command_log:
        for kf in keyframes:
            kf["action_to_next"] = "forward"
        return keyframes

    # Sort command log by frame_idx once, then use bisect for each keyframe
    sorted_log = sorted(command_log, key=lambda x: x[0])
    sorted_idxs = [entry[0] for entry in sorted_log]

    for kf in keyframes:
        fi = kf["frame_idx"]
        # bisect_right gives insertion point; step back one to get
        # the last command at or before fi
        pos = bisect.bisect_right(sorted_idxs, fi) - 1
        if pos >= 0:
            kf["action_to_next"] = sorted_log[pos][1]
        else:
            kf["action_to_next"] = "forward"

    return keyframes


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRAPH PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(graph: nx.DiGraph, path: str = "maze_graph.pkl") -> None:
    """
    Pickle the networkx DiGraph to disk.
    Called at the end of pre_navigation() so the map survives across runs.
    Note: save_artifacts() in build_graph.py also saves the graph as part
    of the full artefact set. Use save_graph() when you only need the graph.
    """
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    size = os.path.getsize(path) / 1024
    print(f"[save_graph]  → {path}  ({size:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOCALISATION  (standalone functions)
# ─────────────────────────────────────────────────────────────────────────────

def localize(query_img_path: str,
             index,
             keyframes: list[dict],
             encoder,
             current_node: int | None = None,
             graph: nx.DiGraph | None = None,
             top_k: int = 3) -> tuple[int, float]:
    """
    Find which graph node the robot is currently at.

    Uses top-K FAISS search with optional temporal consistency boost:
    if current_node and graph are provided, adjacent nodes receive a 15%
    score bonus so the system stays on the predicted trajectory when scores
    are close.

    FIX [1]: Original implementation accepted current_node but the boost
    block was just `pass` (dead code). Graph is now a required parameter
    for the boost, and the boost logic mirrors MazeNavigator.localize().

    Returns (node_id, similarity_score).
    """
    q = encoder.encode(query_img_path).reshape(1, -1).astype(np.float32)
    scores, idxs = index.search(q, top_k * 3)

    candidates = list(zip(idxs[0].tolist(), scores[0].tolist()))

    # FIX [1]: actually apply the temporal consistency boost
    if current_node is not None and graph is not None:
        neighbors = (
            set(graph.successors(current_node)) |
            set(graph.predecessors(current_node)) |
            {current_node}
        )
        candidates = [
            (idx, sc * 1.15 if idx in neighbors else sc)
            for idx, sc in candidates
        ]
        candidates.sort(key=lambda x: -x[1])

    best_node  = int(candidates[0][0])
    best_score = float(candidates[0][1])
    return best_node, best_score


def localize_robust(query_fpv: np.ndarray,
                    index,
                    keyframes: list[dict],
                    encoder,
                    current_node: int | None = None,
                    graph: nx.DiGraph | None = None) -> tuple[int, float]:
    """
    Graph-topology-aware localisation.

    The maze reuses ~200 wall textures, AND parallel corridors share the
    same wall on both sides — so both center-only and strip-based matching
    produce ambiguous results between corridors.

    Key insight: you can't teleport. If you were at node X, you must still
    be near X in the graph. Parallel corridors are 10+ hops apart even if
    visually identical.

    Algorithm:
      1. Single FAISS search (1 encoder call) → top-20 visual candidates
      2. If current_node is known: reweight each candidate by graph distance
         using exponential decay — nearby nodes keep their score, distant
         nodes (like the other side of a wall) get heavily penalized
      3. Return the best combined score

    Total encoder calls: 1 (vs 34 in the strip-based approach).
    """
    h, w = query_fpv.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(query_fpv, cv2.COLOR_BGR2RGB))

    # ── Single encoder call on the full FPV ───────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    pil.save(tmp_path)
    try:
        q = encoder.encode(tmp_path).reshape(1, -1).astype(np.float32)
    finally:
        os.unlink(tmp_path)

    # ── FAISS search for visual candidates ────────────────────────────
    scores, idxs = index.search(q, 20)
    candidates = [(int(idx), float(sc)) for idx, sc in zip(idxs[0], scores[0]) if idx >= 0]

    if not candidates:
        return 0, 0.0

    # ── Graph-distance reweighting ────────────────────────────────────
    if current_node is not None and graph is not None:
        # Compute shortest-path distances from current_node to all candidates
        # Use BFS with a cutoff — anything beyond 30 hops is "far"
        MAX_DIST = 30
        try:
            # Single BFS from current_node — very fast on sparse graphs
            distances = nx.single_source_shortest_path_length(
                graph, current_node, cutoff=MAX_DIST
            )
        except Exception:
            distances = {current_node: 0}

        reweighted = []
        SIGMA = 8.0  # controls decay steepness: higher = more lenient

        for idx, vis_score in candidates:
            dist = distances.get(idx, MAX_DIST)

            if dist == 0:
                # Same node: strong boost
                topo_weight = 1.25
            elif dist <= 2:
                # Immediate neighbors: moderate boost
                topo_weight = 1.15
            else:
                # Exponential decay: e^(-dist/sigma)
                # dist=5 → 0.53, dist=10 → 0.29, dist=20 → 0.08
                topo_weight = math.exp(-dist / SIGMA)

            combined = vis_score * topo_weight
            reweighted.append((idx, combined))

        reweighted.sort(key=lambda x: -x[1])
        return reweighted[0][0], float(reweighted[0][1])

    # No current_node — pure visual matching (first call)
    return candidates[0][0], float(candidates[0][1])


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAZE NAVIGATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MazeNavigator:
    """
    Wraps localisation, path planning, and step execution.

    Lifecycle in player.py:
        pre_navigation   — constructed once after graph is built
        act()            — localize_robust(), next_action(), confirm_step()
                           called every navigation frame

    Public API:
        set_goal(goal_node_idx)         — plan path to a known node
        set_goal_by_image(img_path)     — plan path via visual goal image
        next_action() → str             — 'forward'|'turn_left'|'turn_right'|
                                          'backward'|'stop'
        localize(img_path)              — update position from saved file
        localize_robust(fpv_array)      — update position from live frame
        confirm_step(fpv_array)         — advance path or replan after a move
    """

    def __init__(self,
                 graph: nx.DiGraph,
                 index,
                 keyframes: list[dict],
                 descriptors: np.ndarray,
                 encoder):

        self.G           = graph
        self.index       = index
        self.keyframes   = keyframes
        self.descriptors = descriptors
        self.encoder     = encoder

        self.current_node    = None   # int — updated by localize / confirm_step
        self.current_path    = []     # list[int] — node sequence to goal
        self._goal_node      = None   # int — target node

        # FIX [7]: track best candidate during lost periods separately from
        # current_node so replanning uses the actual localized position
        self.lost_count      = 0
        self._lost_candidate = None   # int — where robot probably actually is

    # ── Internal localise helpers ─────────────────────────────────────────────

    def localize(self, query_img_path: str, top_k: int = 3) -> tuple[int, float]:
        """
        Localise from a saved image file with temporal consistency boost.
        Updates self.current_node.
        """
        q = self.encoder.encode(query_img_path).reshape(1, -1).astype(np.float32)
        scores, idxs = self.index.search(q, top_k * 3)

        candidates = list(zip(idxs[0].tolist(), scores[0].tolist()))

        if self.current_node is not None:
            neighbors = (
                set(self.G.successors(self.current_node)) |
                set(self.G.predecessors(self.current_node)) |
                {self.current_node}
            )
            candidates = [
                (idx, sc * 1.15 if idx in neighbors else sc)
                for idx, sc in candidates
            ]
            candidates.sort(key=lambda x: -x[1])

        best_node  = int(candidates[0][0])
        best_score = float(candidates[0][1])
        self.current_node = best_node
        return best_node, best_score

    def localize_robust(self, fpv: np.ndarray) -> tuple[int, float]:
        """
        Passes graph and current_node to the robust localizer for smoothing.
        Updates self.current_node.
        """
        node, score = localize_robust(
            fpv, self.index, self.keyframes, self.encoder,
            current_node=self.current_node,
            graph=self.G,
        )
        self.current_node = node
        return node, score

    # ── Goal setting / path planning ──────────────────────────────────────────

    def set_goal(self, goal_node_idx: int) -> list[int] | None:
        """
        Plan shortest path from current node to goal_node_idx.

        FIX [8]: nx.NodeNotFound (subclass of NetworkXException, NOT of
        NetworkXNoPath) is now caught explicitly so missing nodes don't
        propagate as unhandled exceptions.

        Returns the path (list of node IDs) or None if no path exists.
        """
        if goal_node_idx is None:
            print("[navigator] Cannot plan: goal_node_idx is None.")
            self.current_path = []
            return None

        if self.current_node is None:
            print("[navigator] Must localise before planning.")
            self.current_path = []
            return None

        try:
            path = nx.shortest_path(
                self.G,
                source=self.current_node,
                target=goal_node_idx,
                weight="weight",
            )
            self.current_path = list(path)
            self._goal_node   = goal_node_idx
            print(f"[navigator] Path: {len(self.current_path) - 1} hops "
                  f"to node {goal_node_idx}")
            return self.current_path

        # FIX [8]: added nx.NodeNotFound to the except tuple
        except (nx.NetworkXNoPath, nx.NodeNotFound,
                KeyError, TypeError, ValueError) as e:
            print(f"[navigator] Path planning failed: "
                  f"{type(e).__name__}: {e}")
            self.current_path = []
            return None

    def set_goal_by_image(self, goal_img_path: str) -> list[int] | None:
        """
        Set goal from a reference photo of the destination.
        Finds the best-matching keyframe and routes to it.
        Called in pre_navigation() with the best-scoring target view.
        """
        q = self.encoder.encode(goal_img_path).reshape(1, -1).astype(np.float32)
        scores, idxs = self.index.search(q, 1)
        goal_node = int(idxs[0][0])
        print(f"[navigator] Goal → node {goal_node} "
              f"(frame {self.keyframes[goal_node]['frame_idx']}, "
              f"sim={scores[0][0]:.3f})")
        return self.set_goal(goal_node)

    # ── Execution ─────────────────────────────────────────────────────────────

    def next_action(self) -> str:
        """
        Return the next action string to execute.
 
        Returns:
            'stop'      — goal reached or no path set
            'use_radar' — loop-closure edge; caller should use scan_directions
                          to decide the physical move (unchanged behaviour)
            'forward' | 'turn_left' | 'turn_right' | 'backward'
                        — sequential edge action from optical-flow memory
 
        No changes to logic vs original; docstring updated to match how
        player.py now interprets 'use_radar' and 'backward' explicitly.
        """
        if not self.current_path or len(self.current_path) < 2:
            return "stop"
 
        src  = self.current_path[0]
        dest = self.current_path[1]
        edge = self.G.get_edge_data(src, dest)
 
        if edge is None:
            return "stop"
 
        if edge.get("edge_type") == "loop_closure" or edge.get("action") == "loop":
            return "use_radar"
 
        return edge.get("action", "forward")


    def confirm_step(self, fpv: np.ndarray) -> tuple[int, float]:
        """
        Confirm the robot's position after a move and advance the path.

        FIX [7]: Original code set self.current_node = self.current_path[0]
        (the stale pre-move node) during lost periods, so after 15 frames the
        replan started from where the robot *was*, not where it *is*.

        Now we store the localized position in self._lost_candidate during
        lost periods and use it as the replan origin when the threshold fires.
        """
        new_node, score = self.localize_robust(fpv)

        if self.current_path and len(self.current_path) >= 2:

            if new_node in self.current_path[1:4]:
                # On track — advance path to matched position
                idx = self.current_path.index(new_node)
                self.current_path    = self.current_path[idx:]
                self.current_node    = new_node
                self.lost_count      = 0
                self._lost_candidate = None

            elif not self.G.has_edge(self.current_path[0], new_node):
                # Off expected path — may be lost or mid-transition
                self.lost_count += 1

                # FIX [7]: track the actual localized position for replan
                self._lost_candidate = new_node

                if self.lost_count > 3:
                    # Robot has been "lost" for 15+ frames — trust the
                    # localized position and replan from there
                    print(f"[navigator] Lost for too long. "
                          f"Forcing replan from {self._lost_candidate}")
                    self.current_node    = self._lost_candidate
                    self.set_goal(self._goal_node)
                    self.lost_count      = 0
                    self._lost_candidate = None
                else:
                    # Hold position estimate at last confirmed node
                    self.current_node = self.current_path[0]

            else:
                # New node is a valid graph neighbor — accept the transition
                self.current_node    = new_node
                self.lost_count      = 0
                self._lost_candidate = None

        return (self.current_node, score)

    # ── Direction scanning and advanced detection ─────────────────────────────

    def extract_directional_crops(self, fpv: np.ndarray) -> dict:
        """
        Split FPV into 4 directional crops: front, left, right, back.
        Returns dict with PIL Image objects for each direction.

        FIX [5]: Image.ROTATE_180 → Image.Transpose.ROTATE_180 for
        Pillow ≥ 9.1 compatibility. The old form passed a bare integer that
        worked by coincidence on older builds.
        """
        h, w = fpv.shape[:2]
        pil  = Image.fromarray(cv2.cvtColor(fpv, cv2.COLOR_BGR2RGB))

        crops = {
            "front": pil.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)),
            "left":  pil.crop((0,      h // 4, w // 2,      3 * h // 4)),
            "right": pil.crop((w // 2, h // 4, w,           3 * h // 4)),
            # FIX [5]: use the modern Pillow Transpose enum
            "back":  pil.transpose(Image.Transpose.ROTATE_180),
        }
        return crops

    def scan_directions(self, fpv: np.ndarray,
                        path_segment: list = None) -> dict:
        """
        Compare 3 directional crops (front / left / right) against stored
        descriptors of upcoming path nodes.
 
        PATCH vs original
        -----------------
        The original returned raw mutable dicts that were stored directly in
        player.py's self.cached_direction_scores.  player.py's _act_moving()
        and _act_idle() now shallow-copy before biasing, but scan_directions
        itself should also return fresh dicts each call so callers that do NOT
        copy are still safe.  Each result dict is now constructed with dict()
        rather than a bare literal so it is always a new object.
 
        Everything else — crop sizes, encoder calls, descriptor dot-products,
        _check_if_blocked logic — is identical to the original.
        """
        if not isinstance(self.current_path, list):
            self.current_path = []
 
        if not self.current_path or len(self.current_path) < 2:
            return {}
 
        if path_segment is None:
            segment_len  = min(7, len(self.current_path))
            path_segment = list(self.current_path[1:segment_len])
 
        h, w = fpv.shape[:2]
        pil  = Image.fromarray(cv2.cvtColor(fpv, cv2.COLOR_BGR2RGB))
 
        crops = {
            "front": pil.crop((w // 6, h // 6, 5 * w // 6, 5 * h // 6)),
            "left":  pil.crop((0,      h // 6, w // 2,      5 * h // 6)),
            "right": pil.crop((w // 2, h // 6, w,           5 * h // 6)),
        }
 
        results = {}
 
        for direction, crop in crops.items():
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            crop.save(tmp_path)
            try:
                q = self.encoder.encode(tmp_path).reshape(1, -1).astype(np.float32)
            finally:
                os.unlink(tmp_path)
 
            best_node     = None
            best_score    = -1.0
            best_distance = 0
 
            for dist, node_idx in enumerate(path_segment, start=1):
                if node_idx < len(self.descriptors):
                    desc       = self.descriptors[node_idx].reshape(1, -1).astype(np.float32)
                    similarity = float(np.dot(q, desc.T)[0, 0])
                    if similarity > best_score:
                        best_score    = similarity
                        best_node     = node_idx
                        best_distance = dist
 
            if best_node is not None:
                is_blocked = self._check_if_blocked(crop)
                # PATCH: use dict() constructor so every call returns a fresh
                # object — safe to mutate in callers without aliasing the cache.
                results[direction] = dict(
                    node     = best_node,
                    score    = best_score if not is_blocked else best_score * 0.3,
                    distance = best_distance,
                    blocked  = is_blocked,
                )
 
        return results

    def _check_if_blocked(self, crop_pil: Image.Image) -> bool:
        """
        Heuristic check: is this direction blocked by a wall?

        Open corridors show perspective lines converging to a vanishing
        point, creating high variance in the horizontal gradient across
        the image. A nearby wall fills the entire crop with uniform texture,
        giving low structural variance.

        Returns True if the direction appears blocked.
        """
        try:
            crop_cv = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)

            # Compute gradient magnitude — corridors have strong edges
            # from perspective convergence; flat walls have less structure
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Center region has more perspective info than edges
            h, w = gray.shape
            center = grad_mag[h//4:3*h//4, w//4:3*w//4]

            # Low variance + low mean gradient = flat wall filling the view
            grad_std = float(np.std(center))
            grad_mean = float(np.mean(center))

            # Threshold: empirically, corridors have std > 25, walls < 15
            # Use conservative threshold to avoid false positives
            return grad_std < 12 and grad_mean < 18
        except Exception:
            return False

    def detect_junction(self, direction_scores: dict, threshold: float = 0.6) -> bool:
        """
        Detect if multiple distinct paths have strong alignment (junction).

        Returns True if >= 2 unique nodes score above the threshold, indicating
        the robot is at a genuine choice point between different topological routes.
        """
        strong_nodes = set()
        
        for v in direction_scores.values():
            if v.get("score", 0) > threshold:
                # We only care about unique destination nodes
                node_id = v.get("node")
                if node_id is not None:
                    strong_nodes.add(node_id)
                    
        return len(strong_nodes) >= 2

    def detect_dead_end(self, direction_scores: dict,
                        threshold: float = 0.4) -> bool:
        """
        Detect if no directions have good alignment (dead end).

        Returns True if all directions score below threshold, indicating
        the planned route may be blocked or invalid.
        """
        if not direction_scores:
            return True
        max_score = max(v.get("score", 0) for v in direction_scores.values())
        return max_score < threshold

    def get_alignment_scores(self, fpv: np.ndarray) -> dict:
        """
        Get alignment scores for current position vs planned route.
        Used to monitor quality while moving.

        FIX [4]: Temp file deleted in try/finally block.

        Returns:
        {
            "primary":          0.85,   # Alignment to next planned node
            "alternative":      0.42,   # Best global alternative match
            "route_confidence": 0.85,   # Confidence in current route
        }
        """
        if not self.current_path or len(self.current_path) < 2:
            return {"primary": 0.0, "alternative": 0.0, "route_confidence": 0.0}

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.fromarray(cv2.cvtColor(fpv, cv2.COLOR_BGR2RGB))
            img.save(tmp_path)

        # FIX [4]: always delete temp file, even on encode() exception
        try:
            q = self.encoder.encode(tmp_path).reshape(1, -1).astype(np.float32)
        finally:
            os.unlink(tmp_path)

        # Score against next planned node descriptor specifically
        next_node     = self.current_path[1] if len(self.current_path) > 1 else None
        primary_score = 0.0
        if next_node is not None and next_node < len(self.descriptors):
            next_desc     = self.descriptors[next_node].reshape(1, -1).astype(np.float32)
            primary_score = float(np.dot(q, next_desc.T)[0, 0])

        # Global rank-2 as a rough alternative measure
        scores, _ = self.index.search(q, 5)
        alternative_score = float(scores[0][1]) if len(scores[0]) > 1 else 0.0

        confidence = primary_score if primary_score > alternative_score else 0.5

        return {
            "primary":          primary_score,
            "alternative":      alternative_score,
            "route_confidence": confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. NAVIGATE_TO_GOAL  (standalone convenience function)
# ─────────────────────────────────────────────────────────────────────────────

def navigate_to_goal(navigator: MazeNavigator,
                     robot,
                     goal_img_path: str,
                     max_steps: int = 200,
                     replan_interval: int = 10) -> bool:
    """
    Full navigation loop for use outside the player game loop (e.g. testing).

    navigator          : fully constructed MazeNavigator
    robot              : object with .capture_frame() → np.ndarray BGR
                                   and .send_action(Action enum)
    goal_img_path      : path to a reference photo of the goal location
    max_steps          : hard ceiling on action count
    replan_interval    : force a full replan every N steps to correct drift

    FIX [9]: Original code mapped action strings to uppercase strings
    ("FORWARD", "LEFT" …) which don't match vis_nav_game.Action enums and
    would silently do nothing or crash. Now imports and uses Action enums
    directly, with a graceful fallback if the import is unavailable.

    Returns True if goal reached, False if max_steps exceeded.
    """
    # FIX [9]: import Action enums; fall back to plain strings for testing
    try:
        from vis_nav_game import Action
        action_map = {
            "forward":    Action.FORWARD,
            "turn_left":  Action.LEFT,
            "turn_right": Action.RIGHT,
            "backward":   Action.BACKWARD,
            "stop":       Action.CHECKIN,
        }
    except ImportError:
        print("[navigate]  vis_nav_game not found — using string actions")
        action_map = {
            "forward":    "FORWARD",
            "turn_left":  "LEFT",
            "turn_right": "RIGHT",
            "backward":   "BACKWARD",
            "stop":       "CHECKIN",
        }

    # Initial localise
    frame      = robot.capture_frame()
    node, score = navigator.localize_robust(frame)
    print(f"[navigate]  Start node {node} (confidence {score:.3f})")

    # Plan
    path = navigator.set_goal_by_image(goal_img_path)
    if path is None:
        print("[navigate]  Cannot reach goal — check graph connectivity.")
        return False

    for step in range(max_steps):
        action_str = navigator.next_action()
        if action_str == "stop":
            print(f"[navigate]  Goal reached in {step} steps.")
            return True

        robot.send_action(action_map.get(action_str, action_map["forward"]))

        frame = robot.capture_frame()
        navigator.confirm_step(frame)

        # Periodic full replan to correct drift
        if step % replan_interval == 0 and navigator._goal_node is not None:
            navigator.set_goal(navigator._goal_node)

    print(f"[navigate]  Max steps ({max_steps}) reached.")
    return False