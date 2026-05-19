"""
auto_nav.py
Autonomous navigation player — full pipeline.

Offline map (built once, cached to disk)
─────────────────────────────────────────
  1. build_vlad_system      SIFT (cached) → K-Means vocab → VLAD vectors → FAISS index
  2. build_clusters         group sequential similar frames into anchor nodes
  3. build_topological_graph  VLAD retrieval → geometric verification (area spread +
                              horizontal constraint) → motion estimation (essential
                              matrix) → cluster-level edges with direction labels +
                              automatic bidirectional inverse edges
  4. save_map_system / load_map_system  FAISS native I/O + pickle metadata + JSON graph

Online localisation (background thread, every ~3 s)
────────────────────────────────────────────────────
  localize_image  VLAD → verify_geometry → cluster-level score → best cluster ID
  find_path       Dijkstra on cluster graph (distance_cost = 1/match_count)
  direction label on next edge → CMD_TURN_LEFT / CMD_TURN_RIGHT written to buffer

FSM (main thread, every frame)
────────────────────────────────
  NORMAL → GLOBAL_TURN ↔ EVASIVE (prior_state) → CONFIRMING → CHECKIN
  MANUAL  (Space toggles; C fires CHECKIN manually)

Key bindings
────────────
  Space     toggle MANUAL ↔ NORMAL
  C         CHECKIN (MANUAL only)
  Arrows    move (MANUAL only)
  Escape    quit
"""

import heapq
import json
import os
import pickle
import sys
import threading
import time
from collections import defaultdict
from enum import Enum, auto

import cv2
import faiss
import numpy as np
import pygame
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import re

from vis_nav_game import Player, Action, Phase


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# VPR / map
NUM_CLUSTERS         = 64
IMG_SIZE             = (320, 240)
TOP_K                = 30
MIN_MATCH_COUNT      = 10
MAX_Y_DIFF           = 40
MIN_CLUSTER_MATCHES  = 15       # SIFT matches needed to keep two frames in same cluster
MIN_MATCH_AREA_RATIO = 0.05     # verified matches must span >= 5% of image area

# Navigation FSM
FLOOR_V_LOW       = 204         # HSV-V threshold for floor isolation
SCAN_FRAC         = 0.40        # bottom fraction of frame used for floor mask
GOOD_THRESH       = 0.35        # centre zone -> confident FORWARD
WEAK_THRESH       = 0.15        # centre zone -> cautious FORWARD
GLOBAL_FWD_THRESH = 0.40        # centre must reach this to exit GLOBAL_TURN
LOST_THRESH       = 0.05        # total floor below this -> EVASIVE
HYSTERESIS_FRAMES = 8           # FORWARD frames before GLOBAL_TURN -> NORMAL

# Background thread
VPR_STRIDE   = 5                # feed every Nth frame to VPR thread
VPR_INTERVAL = 3.0              # seconds between VPR runs

# Navigation & Graph Configs
DOWNSAMPLE_THRESHOLD = 6000
STRIDE               = 10
TEMPORAL_THRESHOLD   = 10
MIN_CLUSTER_MATCHES  = 15

# Target confirmation thresholds
BACK_TARGET_MATCH_THRESH  = 210   # about 75% of 301
FRONT_TARGET_MATCH_THRESH = 175   # about 75% of 256

# Persistence
_HERE      = os.path.dirname(os.path.abspath(__file__))
MAP_FOLDER = os.environ.get("AUTONAV_MAP_FOLDER",
                            os.path.join(_HERE, "map_artifacts"))
DATA_DIR   = os.environ.get("AUTONAV_DATA_DIR",
                            os.path.join(_HERE, "trajectory_data"))

# ══════════════════════════════════════════════════════════════════════════════
# FSM STATES & COMMAND TOKENS
# ══════════════════════════════════════════════════════════════════════════════

class State(Enum):
    NORMAL      = auto()
    GLOBAL_TURN = auto()
    GLOBAL_SETTLE = auto()
    EVASIVE     = auto()
    CONFIRMING  = auto()
    MANUAL      = auto()
    SEARCH_FRONT = auto()

CMD_TURN_LEFT  = "TURN_LEFT"
CMD_TURN_RIGHT = "TURN_RIGHT"
CMD_CHECKIN    = "CHECKIN"

# ══════════════════════════════════════════════════════════════════════════════
# VPR PIPELINE  (offline + online)
# ══════════════════════════════════════════════════════════════════════════════

_sift         = cv2.SIFT_create()
FEATURE_CACHE = {}              # path -> (kp, des)  RAM cache


def _load_resize(path):
    img = cv2.imread(path)
    return cv2.resize(img, IMG_SIZE) if img is not None else None


def get_frame_idx(file_path):
    """Safely extracts the frame number from the filename."""
    filename = os.path.basename(file_path)
    name, _ = os.path.splitext(filename)
    try:
        return int(''.join(filter(str.isdigit, name)))
    except ValueError:
        return -1
    
def get_cached_features(path_or_img):
    """
    Return (kp, des) for a file path (cached to RAM) or a raw numpy frame (not cached).
    Supports both offline map building and online live-frame queries.
    """
    if isinstance(path_or_img, str):
        if path_or_img in FEATURE_CACHE:
            return FEATURE_CACHE[path_or_img]
        img = _load_resize(path_or_img)
        key = path_or_img
    else:
        img = cv2.resize(path_or_img, IMG_SIZE)
        key = None                              # never cache live frames

    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = _sift.detectAndCompute(gray, None)

    if key is not None:
        FEATURE_CACHE[key] = (kp, des)
    return kp, des

def fast_sift_match(path_a, path_b):
    """Now runs instantly by pulling features from the cache."""
    kp_a, des_a = get_cached_features(path_a)
    kp_b, des_b = get_cached_features(path_b)
    
    if des_a is None or des_b is None: return 0
        
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k=2)
    
    # Safe unpacking for knnMatch
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
    return len(good)


def compute_vlad(des, kmeans):
    """Power-normalised, L2-normalised VLAD descriptor."""
    centers = kmeans.cluster_centers_
    k       = centers.shape[0]
    vlad    = np.zeros((k, centers.shape[1]), dtype=np.float32)
    labels  = kmeans.predict(des)
    for i, d in enumerate(des):
        vlad[labels[i]] += d - centers[labels[i]]
    vlad = vlad.flatten()
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))   # power norm
    norm = np.linalg.norm(vlad)
    if norm > 0:
        vlad /= norm
    return vlad


# ── Offline: map building ──────────────────────────────────────────────────

def build_vlad_system(image_paths):
    """SIFT (cached) -> K-Means vocab -> VLAD vectors -> FAISS index."""
    all_des = []
    for p in tqdm(image_paths, desc="1. Feature extraction"):
        _, des = get_cached_features(p)
        if des is not None:
            all_des.append(des)

    print(f"[MAP] 2. Clustering {len(all_des)} descriptor sets -> {NUM_CLUSTERS} words...")
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=10000, n_init="auto")
    kmeans.fit(np.vstack(all_des))

    vectors, valid_paths = [], []
    for p in tqdm(image_paths, desc="3. Building VLAD index"):
        _, des = get_cached_features(p)
        if des is not None:
            vectors.append(compute_vlad(des, kmeans))
            valid_paths.append(p)

    vectors = np.array(vectors, dtype="float32")
    index   = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    print(f"[MAP] FAISS index: {len(valid_paths)} images.")
    return kmeans, index, valid_paths


def build_clusters(paths):
    print("\n[INFO] Grouping images into Anchor-Based Neighborhood Clusters...")
    clusters, img_to_cluster = {}, {}
    cluster_idx = 0
    
    paths_by_folder = defaultdict(list)
    for p in paths: 
        paths_by_folder[os.path.dirname(p)].append(p)
        
    for folder, f_paths in paths_by_folder.items():
        f_paths = sorted(f_paths, key=get_frame_idx)
        
        curr_cluster = [f_paths[0]]
        clusters[cluster_idx] = curr_cluster
        img_to_cluster[f_paths[0]] = cluster_idx
        
        for i in tqdm(range(1, len(f_paths)), desc=f"Clustering {os.path.basename(folder)}"):
            curr_path = f_paths[i]
            anchor_path = curr_cluster[0] 
            
            # 1. Temporal Check against Anchor
            if get_frame_idx(curr_path) - get_frame_idx(anchor_path) <= TEMPORAL_THRESHOLD:
                # 2. Visual Check against Anchor
                if fast_sift_match(anchor_path, curr_path) >= MIN_CLUSTER_MATCHES:
                    curr_cluster.append(curr_path)
                    img_to_cluster[curr_path] = cluster_idx
                    continue
            
            cluster_idx += 1
            curr_cluster = [curr_path]
            clusters[cluster_idx] = curr_cluster
            img_to_cluster[curr_path] = cluster_idx
            
        cluster_idx += 1 

    print(f"[INFO] Compressed {len(paths)} images down to {len(clusters)} Anchor Nodes.")
    return clusters, img_to_cluster

def _estimate_motion(kp_q, kp_r, matches):
    """
    Estimate relative camera motion via essential matrix (RANSAC).
    Returns dict with 'direction_str' or None if estimation fails.
    Directions: Forward / Backward / Left / Right / Minimal
    """
    if len(matches) < 5:
        return None

    pts_q = np.float32([kp_q[m.queryIdx].pt for m in matches])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in matches])

    # Approximate intrinsics for IMG_SIZE (320x240)
    K = np.array([[92, 0, 160],
                  [0,  92, 120],
                  [0,   0,   1]], dtype=np.float32)

    E, _ = cv2.findEssentialMat(pts_q, pts_r, K,
                                 method=cv2.RANSAC, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None

    _, _, t, _ = cv2.recoverPose(E, pts_q, pts_r, K)
    x, _, z    = t.flatten()

    if np.linalg.norm([x, z]) < 0.15:
        return {"direction_str": "Minimal"}
    if abs(x) > abs(z):
        direction = "Right" if x > 0 else "Left"
    else:
        direction = "Forward" if z > 0 else "Backward"
    return {"direction_str": direction}


def query_vlad(path_or_img, kmeans, index, paths):
    """VLAD query -- accepts a file path or a raw numpy frame."""
    _, des = get_cached_features(path_or_img)
    if des is None:
        return []
    vlad = compute_vlad(des, kmeans).astype("float32").reshape(1, -1)
    D, I = index.search(vlad, TOP_K)
    return [(paths[i], D[0][j]) for j, i in enumerate(I[0])]


def verify_geometry(path_or_img, vlad_results):
    """
    Geometric verification with:
      - Lowe's ratio test
      - Horizontal constraint (MAX_Y_DIFF)
      - Spatial area spread check (MIN_MATCH_AREA_RATIO)

    Accepts a file path or raw numpy frame as query.
    Returns (verified_results, kp_q).
    verified_results items include 'kp_r' and 'good_matches' for motion estimation.
    """
    kp_q, des_q = get_cached_features(path_or_img)
    if des_q is None:
        return [], None

    bf         = cv2.BFMatcher()
    total_area = IMG_SIZE[0] * IMG_SIZE[1]
    verified   = []

    for path, _ in vlad_results:
        kp_r, des_r = get_cached_features(path)
        if des_r is None:
            continue

        matches = bf.knnMatch(des_q, des_r, k=2)
        good    = [m_n[0]  for m_n in matches
                   if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance]

        # Horizontal constraint
        valid = [m for m in good
                 if abs(kp_q[m.queryIdx].pt[1] - kp_r[m.trainIdx].pt[1]) <= MAX_Y_DIFF]

        if len(valid) < MIN_MATCH_COUNT:
            continue

        # Area spread check -- reject tiny repetitive textures
        pts_q      = np.float32([kp_q[m.queryIdx].pt for m in valid])
        x_min, y_min = np.min(pts_q, axis=0)
        x_max, y_max = np.max(pts_q, axis=0)
        area_ratio   = (x_max - x_min) * (y_max - y_min) / total_area
        if area_ratio < MIN_MATCH_AREA_RATIO:
            continue

        verified.append({
            "path":         path,
            "match_count":  len(valid),
            "kp_r":         kp_r,
            "good_matches": valid,
        })

    verified.sort(key=lambda x: x["match_count"], reverse=True)
    return verified, kp_q


def build_topological_graph(kmeans, index, paths, action_map, clusters, img_to_cluster):
    print("\n[INFO] Building Spatial-Aware Topological Graph...")
    
    INVERSE = {
        "Forward": "Backward", "Backward": "Forward",
        "Left": "Right", "Right": "Left",
        "Unknown": "Unknown", "Minimal": "Minimal", "Idle": "Idle"
    }
    
    temp_edges = defaultdict(lambda: defaultdict(lambda: {
        "direction_weights": defaultdict(float),
        "total_weight": 0, "vote_count": 0, "is_sequential": False
    }))
    
    for path in tqdm(paths, desc="Mapping Connections"):
        cluster_a = img_to_cluster[path]
        folder_a = os.path.dirname(path)
        idx_a = get_frame_idx(path)
        
        vlad_results = query_vlad(path, kmeans, index, paths)[:20]
        verified_results, kp_q = verify_geometry(path, vlad_results)
        
        for res in verified_results:
            neighbor_path = res["path"]
            cluster_b = img_to_cluster[neighbor_path]
            
            if cluster_a == cluster_b: continue
                
            folder_b = os.path.dirname(neighbor_path)
            idx_b = get_frame_idx(neighbor_path)
            weight = res["match_count"]
            direction = "Unknown"
            
            is_temporally_local = ((folder_a == folder_b) and idx_a != -1 and idx_b != -1 and abs(idx_a - idx_b) <= TEMPORAL_THRESHOLD)

            # Inject Ground Truth or fallback to math
            if is_temporally_local and path in action_map:
                direction = action_map[path]
                if direction.upper() in ["IDLE", "UNKNOWN"]:
                    motion = _estimate_motion(kp_q, res["kp_r"], res["good_matches"])
                    if motion: direction = motion["direction_str"]
            else:
                motion = _estimate_motion(kp_q, res["kp_r"], res["good_matches"])
                if motion: direction = motion["direction_str"]
            
            if is_temporally_local:
                temp_edges[cluster_a][cluster_b]["is_sequential"] = True
                
            temp_edges[cluster_a][cluster_b]["direction_weights"][direction] += weight
            temp_edges[cluster_a][cluster_b]["total_weight"] += weight
            temp_edges[cluster_a][cluster_b]["vote_count"] += 1

    final_graph = defaultdict(dict)
    
    for c_a, neighbors in temp_edges.items():
        for c_b, data in neighbors.items():
            if data["vote_count"] == 0: continue 
                
            valid_dirs = {k: v for k, v in data["direction_weights"].items() if k not in ["Unknown", "Minimal"]}
            if valid_dirs:
                winning_direction = max(valid_dirs, key=valid_dirs.get)
            else:
                winning_direction = max(data["direction_weights"], key=data["direction_weights"].get) 
                
            inverse_dir = INVERSE.get(winning_direction, "Unknown")
            
            avg_weight = data["total_weight"] / data["vote_count"]
            base_cost = 1.0 / (avg_weight + 0.001)
            
            # Loop closure penalty
            distance_cost = base_cost * (1.0 if data["is_sequential"] else 3.0) 
            
            str_ca, str_cb = str(c_a), str(c_b)
            
            final_graph[str_ca][str_cb] = {
                "weight": avg_weight,
                "distance_cost": distance_cost,
                "direction": winning_direction,
                "type": "sequential" if data["is_sequential"] else "loop_closure"
            }
            
            if str_ca not in final_graph.get(str_cb, {}):
                final_graph[str_cb][str_ca] = {
                    "weight": avg_weight,
                    "distance_cost": distance_cost,
                    "direction": inverse_dir,
                    "type": "sequential" if data["is_sequential"] else "loop_closure"
                }
            
    return dict(final_graph)

# ── Persistence ────────────────────────────────────────────────────────────

def save_map_system(kmeans, index, valid_paths, clusters, img_to_cluster,
                    graph, folder=MAP_FOLDER):
    """
    Save all pipeline components.
    - metadata.pkl      : KMeans + paths + clusters (must stay in sync)
    - vlad.index        : FAISS native format (faster/smaller than pickle)
    - topological_graph.json : human-readable graph for debugging
    """
    os.makedirs(folder, exist_ok=True)

    metadata = {
        "kmeans":         kmeans,
        "valid_paths":    valid_paths,
        "clusters":       clusters,
        "img_to_cluster": img_to_cluster,
    }
    with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    faiss.write_index(index, os.path.join(folder, "vlad.index"))

    with open(os.path.join(folder, "topological_graph.json"), "w") as f:
        json.dump(graph, f, indent=4)

    print(f"[MAP] Saved to '{folder}/'")


def load_map_system(folder=MAP_FOLDER):
    """Load all pipeline components from disk."""
    print(f"[MAP] Loading from '{folder}'...")

    with open(os.path.join(folder, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    index = faiss.read_index(os.path.join(folder, "vlad.index"))

    with open(os.path.join(folder, "topological_graph.json"), "r") as f:
        graph = json.load(f)

    print(f"[MAP] Loaded: {len(meta['valid_paths'])} images, "
          f"{len(meta['clusters'])} clusters, {len(graph)} graph nodes.")
    return (
        meta["kmeans"],
        index,
        meta["valid_paths"],
        meta["clusters"],
        meta["img_to_cluster"],
        graph,
    )


# ── Online localisation ────────────────────────────────────────────────────

def localize_image(cv2_img, kmeans, index, paths, img_to_cluster):
    """
    Real-time localisation entry point.
    Accepts a raw BGR numpy frame.
    Returns best cluster ID (int) or None.
    """
    vlad_res        = query_vlad(cv2_img, kmeans, index, paths)
    verified, _     = verify_geometry(cv2_img, vlad_res)
    if not verified:
        return None

    scores = defaultdict(float)
    for res in verified:
        c_id          = img_to_cluster[res["path"]]
        scores[c_id] += res["match_count"]
    return max(scores, key=scores.get)


def find_path(graph, start_node, target_node):
    """
    Dijkstra on the cluster graph.
    Nodes are string keys; uses 'distance_cost' (= 1/match_count) as edge weight.
    Returns ordered list [start, ..., target].
    Returns [] if start == target.
    """
    start, target = str(start_node), str(target_node)
    if start == target:
        return []

    dist = {start: 0.0}
    prev = {}
    pq   = [(0.0, start)]

    while pq:
        d, curr = heapq.heappop(pq)
        if curr == target:
            break
        if d > dist.get(curr, float("inf")):
            continue
        for neighbour, data in graph.get(curr, {}).items():
            nd = d + data["distance_cost"]
            if nd < dist.get(neighbour, float("inf")):
                dist[neighbour] = nd
                prev[neighbour] = curr
                heapq.heappush(pq, (nd, neighbour))

    path, curr = [], target
    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    path.append(start)
    path.reverse()
    return path   # [start, ..., target]

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED MAP STATE
# ──────────────────────────────────────────────────────────────────────────────
class MapState:
    def __init__(self):
        self._lock       = threading.Lock()
        self.node        = None
        self.path        = None
        self.target_done = False

    def update(self, node, path, target_done):
        with self._lock:
            self.node        = node
            self.path        = path
            self.target_done = target_done

    def snapshot(self):
        with self._lock:
            return {
                "node":        self.node,
                "path":        list(self.path) if self.path else None,
                "target_done": self.target_done,
            }

class DeadReckoner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_steps  = 0
        self.age_seconds    = 0.0
        self._last_tick     = time.time()

    def tick(self, action):
        now = time.time()
        self.age_seconds  += now - self._last_tick
        self._last_tick    = now
        if action == Action.FORWARD:
            self.forward_steps += 1
        elif action == Action.BACKWARD:
            self.forward_steps -= 1

    @property
    def confidence(self):
        time_decay = max(0.0, 1.0 - self.age_seconds / 15.0)
        step_decay = max(0.0, 1.0 - abs(self.forward_steps) / 20.0)
        return time_decay * step_decay

def sliding_cursor(path, current_node):
    if not path or current_node is None:
        return None
    for i, node in enumerate(path):
        if node == str(current_node):
            return i
    return None

# ══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS NAVIGATION PLAYER
# ══════════════════════════════════════════════════════════════════════════════

class AutoNavPlayer(Player):
    """
    Full autonomous navigation player.

    FSM: NORMAL | GLOBAL_TURN | EVASIVE | CONFIRMING | MANUAL
    Keys: Space = MANUAL toggle, C = CHECKIN (MANUAL), Arrows = move (MANUAL), Esc = quit
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __init__(self):
        # ── Visual & Threading ──
        self.fpv = None
        self.screen = None
        self._mode = "auto"
        self._quit = False
        self.last_act = Action.IDLE
        self._frame_count = 0

        # ── FSM & Navigation ──
        self.fsm_state = State.NORMAL
        self._prior_state = State.NORMAL
        self.global_cmd = None
        self._settle_frames = 0
        self._local_turning = None
        self._target_front_img = None
        self._target_back_img = None
        self._target_node = None

        # ── Tracking & Dead Reckoning ──
        self._map_state = MapState()
        self._dr = DeadReckoner()
        self._snap = {"node": None, "path": None, "target_done": False}
        self._edge_consumed = None
        
        # ── Background Communication ──
        self._frame_queue = None
        self._fq_lock = threading.Lock()

        # Map artefacts placeholders
        self._kmeans = None
        self._faiss_idx = None
        self._map_paths = None
        self._clusters = None
        self._img_to_cluster = None
        self._graph = None

        super().__init__()
        print("[AutoNav] Refined v2-Architecture Initialised.")

    def reset(self):
        pygame.init()
        self.fsm_state = State.NORMAL
        self.global_cmd = None
        self._settle_frames = 0
        self._local_turning = None
        self._mode = "auto"
        self._dr.reset()
        self._edge_consumed = None
        self.last_act = Action.IDLE
        print("[AutoNav] Reset complete: FSM cleared.")
        self.fpv      = None
        self.screen   = None
        self.last_act = Action.IDLE
        pygame.init()
        print("[AutoNav] Reset complete.")

    # ── pre_exploration ───────────────────────────────────────────────────────

    def pre_exploration(self):
        """
        Build or load the map before the game starts.
        act() returns QUIT immediately when the engine enters Phase.EXPLORATION.
        """
        print("[AutoNav] pre_exploration: building / loading map...")
        self._build_or_load_map()
        print("[AutoNav] pre_exploration done. Will QUIT exploration phase immediately.")

    def _build_or_load_map(self):
        map_exists = (
            os.path.exists(MAP_FOLDER)
            and os.path.exists(os.path.join(MAP_FOLDER, "metadata.pkl"))
            and os.path.exists(os.path.join(MAP_FOLDER, "vlad.index"))
            and os.path.exists(os.path.join(MAP_FOLDER, "topological_graph.json"))
        )

        if map_exists:
            (self._kmeans, self._faiss_idx, self._map_paths,
             self._clusters, self._img_to_cluster, self._graph) = load_map_system()
            return

        all_paths = []
        action_map = {}

        if os.path.isdir(DATA_DIR):
            subdirs = sorted([os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) 
                              if os.path.isdir(os.path.join(DATA_DIR, d))])
            
            for folder in subdirs:
                folder_name = os.path.basename(folder)
                
                # 1. Load Actions (JSON)
                json_path = os.path.join(folder, f"{folder_name}.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            img_name = item.get("image")
                            action = item.get("action", ["UNKNOWN"])[0].title()
                            full_path = os.path.abspath(os.path.join(folder, img_name))
                            action_map[full_path] = action

                # 2. Load Images with Dynamic Downsampling
                paths = sorted([
                    os.path.abspath(os.path.join(folder, f))
                    for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                
                num_images = len(paths)
                if num_images > DOWNSAMPLE_THRESHOLD:
                    print(f"[INFO] {folder_name} has {num_images} images. Downsampling (Stride {STRIDE})...")
                    sampled_paths = paths[::STRIDE]
                else:
                    sampled_paths = paths
                    
                all_paths.extend(sampled_paths)
        else:
            print(f"[AutoNav] WARNING: DATA_DIR '{DATA_DIR}' not found.")
            return

        print(f"[AutoNav] Building map from {len(all_paths)} carefully sampled images...")
        self._map_paths = all_paths
        
        # Build Pipeline
        self._kmeans, self._faiss_idx, self._map_paths = build_vlad_system(self._map_paths)
        
        self._clusters, self._img_to_cluster = build_clusters(self._map_paths)
        
        self._graph = build_topological_graph(
            self._kmeans, self._faiss_idx, self._map_paths, 
            action_map, self._clusters, self._img_to_cluster
        )
        
        save_map_system(
            self._kmeans, self._faiss_idx, self._map_paths,
            self._clusters, self._img_to_cluster, self._graph
        )
        
        # Global calculation speed optimisation: Free RAM for gameplay
        global FEATURE_CACHE
        FEATURE_CACHE.clear()
        print("[AutoNav] Map built. Feature Cache cleared.")

    # ── pre_navigation ────────────────────────────────────────────────────────

    def pre_navigation(self):
        print("[AutoNav] pre_navigation...")

        # 1. Target images
        images = self.get_target_images()
        self.flag = images
        if not images:
            print("[AutoNav] WARNING: no target images.")
            return
        self.show_target_images()

        # Target image convention:
        # images[0] = front target
        # images[2] = back target
        self._target_front_img = images[0].copy()

        if len(images) >= 3:
            self._target_back_img = images[2].copy()
        else:
            print("[AutoNav] WARNING: expected images[2] for back target, but not enough target images.")
            self._target_back_img = None

        # 2. Identify target_node using the BACK target first
        if self._kmeans is not None and self._target_back_img is not None:
            self._target_node = localize_image(
                self._target_back_img,
                self._kmeans,
                self._faiss_idx,
                self._map_paths,
                self._img_to_cluster
            )
            print(f"[AutoNav] BACK target_node = cluster {self._target_node}")
        # 3. Initialise FSM
        self.fsm_state = State.NORMAL
        self._heading = 0
        self._dr.reset()

        # 4. Localise start + initial Dijkstra path
        if self.fpv is not None and self._kmeans is not None and self._target_node is not None:
            start = localize_image(
                self.fpv, self._kmeans, self._faiss_idx,
                self._map_paths, self._img_to_cluster)
            
            if start is not None:
                path = find_path(self._graph, start, self._target_node)
                # v2 fix: push to map_state instead of writing a command
                self._map_state.update(start, path, False)

        # 5. Launch background thread
        t = threading.Thread(target=self._vpr_loop, daemon=True)
        t.start()
        print("[AutoNav] Background VPR thread started.")

    # ── Background VPR / planning thread ─────────────────────────────────────

    def _vpr_loop(self):
        """Every VPR_INTERVAL seconds: localise -> re-plan Dijkstra -> write to MapState."""
        while True:
            time.sleep(VPR_INTERVAL)

            with self._fq_lock:
                frame = self._frame_queue
            if frame is None or self._kmeans is None:
                continue

            cluster = localize_image(
                frame, self._kmeans, self._faiss_idx,
                self._map_paths, self._img_to_cluster)
            
            if cluster is None:
                continue

            # Reached goal flag
            if self._target_node is not None and cluster == self._target_node:
                self._map_state.update(cluster, None, True)
                continue

            # Re-plan path
            if self._target_node is not None:
                path = find_path(self._graph, cluster, self._target_node)
                self._map_state.update(cluster, path, False)

    def _next_edge_direction(self):
        """Reads MapState to determine next turn. Handles drift and consumed edges."""
        snap = self._snap
        if not snap["path"] or snap["node"] is None:
            return None

        cursor = sliding_cursor(snap["path"], snap["node"])
        if cursor is None or cursor >= len(snap["path"]) - 1:
            return None

        src, dst = snap["path"][cursor], snap["path"][cursor + 1]
        edge_key = (src, dst)

        # Gate by consumed edges or low dead-reckoning confidence
        if self._edge_consumed == edge_key or self._dr.confidence < 0.20:
            return None

        edge = self._graph.get(str(src), {}).get(str(dst), {})
        direction = edge.get("direction", "Unknown")
        self._edge_consumed = edge_key

        if direction == "Left": return Action.LEFT
        if direction == "Right": return Action.RIGHT
        return None

    # ── see() ─────────────────────────────────────────────────────────────────

    def see(self, fpv):
        if fpv is None or fpv.ndim < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("AutoNav: FPV")

        rgb  = fpv[:, :, ::-1]
        surf = pygame.image.frombuffer(rgb.tobytes(), (fpv.shape[1], fpv.shape[0]), "RGB")
        self.screen.blit(surf, (0, 0))
        pygame.display.update()

        self._floor_mask = self._get_floor_mask(fpv)

        self._frame_count += 1
        if self._frame_count % VPR_STRIDE == 0:
            with self._fq_lock:
                self._frame_queue = fpv.copy()

    # ── Perception helpers ────────────────────────────────────────────────────

    def _get_floor_mask(self, frame):
        """HSV-V threshold -> morphological clean -> binary mask (bottom SCAN_FRAC)."""
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask   = (hsv[:, :, 2] > FLOOR_V_LOW).astype(np.uint8) * 255
        h      = mask.shape[0]
        crop_y = int(h * (1.0 - SCAN_FRAC))
        roi    = mask[crop_y:, :]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        roi    = cv2.morphologyEx(roi, cv2.MORPH_OPEN,  kernel)
        roi    = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        result          = np.zeros_like(mask)
        result[crop_y:] = roi
        return result

    def _compute_zone_scores(self, mask):
        """Split floor mask into L / C / R thirds -> fraction of white pixels each."""
        h, w   = mask.shape
        crop_y = int(h * (1.0 - SCAN_FRAC))
        roi    = mask[crop_y:, :]
        if roi.size == 0:
            return 0.0, 0.0, 0.0
        third = w // 3
        rh    = roi.shape[0]
        L = np.count_nonzero(roi[:, :third])         / (rh * third)
        C = np.count_nonzero(roi[:, third:2*third])  / (rh * third)
        R = np.count_nonzero(roi[:, 2*third:])       / (rh * (w - 2*third))
        return L, C, R

    # ── act() — FSM main loop ─────────────────────────────────────────────────

    def act(self):
        # 1. THE SKIP TRIGGER (With the frame=0 safety catch)
        if hasattr(self, '_state') and self._state is not None:
            current_phase = self._state[1]
            frame_num = self._state[2] # Extract the current frame number
            
            if current_phase == Phase.EXPLORATION:
                # ONLY quit on the very first frame to prevent double-quitting
                if frame_num == 0:
                    print("[v1] Skipping Exploration: Map already loaded from dataset.")
                    return Action.QUIT
                else:
                    # If the engine is lagging on the transition, just wait safely.
                    return Action.IDLE

        if self.fpv is None: 
            return Action.IDLE

        # 1. Handle UI and Manual Mode
        pump_res = self._pump_events()
        if self._quit: return Action.QUIT
        if pump_res == Action.CHECKIN: return Action.CHECKIN
        
        if self._mode == "manual":
            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                return Action.FORWARD
            if keys[pygame.K_DOWN]:
                return Action.BACKWARD
            if keys[pygame.K_LEFT]:
                return Action.LEFT
            if keys[pygame.K_RIGHT]:
                return Action.RIGHT

            return Action.IDLE

        # 2. Localize & Map Update (Snapshots)
        new_snap = self._map_state.snapshot()
        if new_snap["node"] != self._snap["node"] and new_snap["node"] is not None:
            print(f"[NAV] Node {self._snap['node']} -> {new_snap['node']}. Resetting DR.")
            self._dr.reset()
            self._edge_consumed = None
        self._snap = new_snap

        # 3. Floor Analysis
        mask = self._get_floor_mask(self.fpv)
        L, C, R = self._compute_zone_scores(mask)
        
        # 4. State Transitions (The 'Robust' logic)

        # Throttled Heartbeat (every 10 frames)
        if self._frame_count % 10 == 0:
            path_len = len(self._snap['path']) if self._snap['path'] else 0
            conf = self._dr.confidence
            print(f"[STATUS] State: {self.fsm_state.name} | Sensors: L:{L:.2f} C:{C:.2f} R:{R:.2f} | "
                  f"Node: {self._snap['node']} | Path: {path_len} steps | DR-Conf: {conf:.2f}")
        
        # Priority 1: Obstacle Safety
        if (L + C + R) < LOST_THRESH:
            if self.fsm_state != State.EVASIVE:
                print("[FSM] OBSTACLE -> EVASIVE")
                self._prior_state = self.fsm_state
                self.fsm_state = State.EVASIVE

        # Priority 2: Reached BACK target node
        if self._snap["target_done"] and self.fsm_state not in [State.CONFIRMING, State.SEARCH_FRONT]:
            print("[FSM] BACK target node reached -> CONFIRMING BACK VIEW")
            self.fsm_state = State.CONFIRMING
            return self._confirming_act()


        # Priority 3: Global Navigation
        if self.fsm_state == State.NORMAL:
            cmd = self._next_edge_direction()
            if cmd in [Action.LEFT, Action.RIGHT]:
                self.fsm_state = State.GLOBAL_TURN
                self.global_cmd = cmd
                self._local_turning = None  # Clear local wander state
                # Clear and meaningful message
                print(f"!!! [PLANNER] New Direction Needed: {cmd.name} !!!")
                print(f"[FSM] NORMAL -> GLOBAL_TURN")

       
        # 5. Behavior Dispatch
        # 5. Behavior Dispatch
        if self.fsm_state == State.CONFIRMING:
            return self._confirming_act()

        if self.fsm_state == State.SEARCH_FRONT:
            return self._search_front_act()
        
        if self.fsm_state == State.EVASIVE:
            # If we finally see floor, recover
            if (L + C + R) > WEAK_THRESH:
                print(f"--- [RECOVERY] Floor found ({L+C+R:.2f})! Returning to {self._prior_state.name} ---")
                self.fsm_state = self._prior_state
                return Action.FORWARD

            # If totally blind (like in your log), BACK UP or SPIN HARD
            if (L + C + R) < 0.02:
                # Every 5 frames, try to backup to find perspective, otherwise spin
                if self._frame_count % 5 == 0:
                    return Action.BACKWARD
                return Action.LEFT # Hard spin to find floor

            # Otherwise, turn toward the "least bad" side
            return Action.LEFT if L > R else Action.RIGHT
        
        if self.fsm_state == State.GLOBAL_TURN:
            if C >= GLOBAL_FWD_THRESH:
                self.fsm_state = State.GLOBAL_SETTLE
                self._settle_frames = HYSTERESIS_FRAMES
                print("[FSM] TURN DONE -> SETTLING")
                return Action.FORWARD
            return self.global_cmd

        if self.fsm_state == State.GLOBAL_SETTLE:
            self._settle_frames -= 1
            if self._settle_frames <= 0:
                self.fsm_state = State.NORMAL
                print("[FSM] SETTLE DONE -> NORMAL")
            return Action.FORWARD if C > WEAK_THRESH else (Action.LEFT if L > R else Action.RIGHT)

        # Default Behavior: NORMAL (Wander)
        # Fetch the bias from the global map
        current_bias = self._get_path_bias()
        
        # Pass it into the local planner
        action = self._wander(L, C, R, mask, global_bias=current_bias)
        
        self._dr.tick(action)
        return action
    # ── Event pump ────────────────────────────────────────────────────────────

    def _pump_events(self):
        """v2-style event pump for mode switching and manual override."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit = True
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit = True
                    return Action.QUIT

                if event.key == pygame.K_SPACE:
                    self._mode = "manual" if self._mode == "auto" else "auto"
                    self.last_act = Action.IDLE
                    self._local_turning = None # Clear AI memory on toggle
                    print(f"--- MODE: {self._mode.upper()} ---")

                if self._mode == "manual":
                    if event.key == pygame.K_UP:    self.last_act = Action.FORWARD
                    if event.key == pygame.K_DOWN:  self.last_act = Action.BACKWARD
                    if event.key == pygame.K_LEFT:  self.last_act = Action.LEFT
                    if event.key == pygame.K_RIGHT: self.last_act = Action.RIGHT
                    if event.key == pygame.K_c:     return Action.CHECKIN

        return None
    """
    # ── Obstacle check ────────────────────────────────────────────────────────

    def _check_obstacle(self, total_floor):
        Enter EVASIVE (saving prior_state) if floor coverage drops below threshold.
        if total_floor < LOST_THRESH and self.fsm_state  != State.EVASIVE:
            self._prior_state = self.fsm_state 
            self.fsm_state        = State.EVASIVE
            return True
        return self.fsm_state  == State.EVASIVE

    # ── Global command reader (Fix 4: atomic read+clear) ─────────────────────

    def _check_global_command(self):
        Atomically read and clear the buffer at the moment of read -- not after.
        Transitions state to GLOBAL_TURN or CONFIRMING.
        Returns True if a command was consumed.
        
        with self._buf_lock:
            cmd              = self._cmd_buffer
            self._cmd_buffer = None     # atomic clear -- Fix 4

        if cmd == CMD_TURN_LEFT:
            self._turn_dir  = Action.LEFT
            self._fwd_count = 0
            self.fsm_state      = State.GLOBAL_TURN
            return True

        if cmd == CMD_TURN_RIGHT:
            self._turn_dir  = Action.RIGHT
            self._fwd_count = 0
            self.fsm_state      = State.GLOBAL_TURN
            return True

        if cmd == CMD_CHECKIN:
            self.fsm_state  = State.CONFIRMING
            return True

        return False
    """
    # ── Local planner (NORMAL) ────────────────────────────────────────────────
        
    def _do_settle(self, L, C, R):
        """Hysteresis dwell after a global turn to prevent oscillation."""
        self._settle_frames -= 1
        if self._settle_frames <= 0:
            self.fsm_state = State.NORMAL # Changed from self.fsm_state 
            self._local_turning = None

        if C > GOOD_THRESH or abs(L - R) < 0.05:
            action = Action.FORWARD
        else:
            action = Action.LEFT if L > R else Action.RIGHT
            
        self._update_heading(action)
        return action
   
    def _wander(self, L, C, R, mask, global_bias=None):
        """
        Local hallway steering using:
        1. floor coverage safety
        2. global map bias from Dijkstra path
        3. vanishing-point / floor-centroid steering
        4. fallback zone steering

        L, C, R are floor scores from left/center/right thirds.
        mask is the binary floor mask.
        global_bias can be "Left", "Right", "Forward", or None.
        """

        h, w = mask.shape

        # Use actual floor coverage from the whole mask.
        # This is the vanishing-point steering feature extracted from your other wander function.
        total = np.count_nonzero(mask) / mask.size

        # ─────────────────────────────────────────────
        # 1. Soft global bias from map/Dijkstra planner
        # ─────────────────────────────────────────────
        BIAS_STRENGTH = 0.15
        if global_bias == "Left":
            L += BIAS_STRENGTH
        elif global_bias == "Right":
            R += BIAS_STRENGTH

        # ─────────────────────────────────────────────
        # 2. Emergency recovery if floor is mostly gone
        # ─────────────────────────────────────────────
        if total < LOST_THRESH:
            self._local_turning = "LEFT" if L > R else "RIGHT"
            return Action.LEFT if self._local_turning == "LEFT" else Action.RIGHT
        # ─────────────────────────────────────────────
        # 3. Vanishing point / hallway center estimate
        # ─────────────────────────────────────────────
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
        else:
            cx = w // 2

        target_x = w // 2
        margin = int(w * 0.15)

        if self._frame_count % 20 == 0:
            print(
                f"[VANISHING] cx={cx}, target={target_x}, margin={margin}, "
                f"bias={global_bias}, L:{L:.2f} C:{C:.2f} R:{R:.2f}, floor={total:.2f}"
            )

        # ─────────────────────────────────────────────
        # 4. Sticky logic: finish turns once hallway center is aligned
        # ─────────────────────────────────────────────
        if self._local_turning is not None:
            if C > GOOD_THRESH and abs(cx - target_x) <= margin:
                self._local_turning = None
                return Action.FORWARD

            return Action.LEFT if self._local_turning == "LEFT" else Action.RIGHT

        # ─────────────────────────────────────────────
        # 5. Vanishing-point steering
        # Keep the floor centroid centered in the image.
        # ─────────────────────────────────────────────
        if cx < target_x - margin:
            self._local_turning = "LEFT"
            return Action.LEFT

        if cx > target_x + margin:
            self._local_turning = "RIGHT"
            return Action.RIGHT

        # ─────────────────────────────────────────────
        # 6. If centered, drive forward
        # ─────────────────────────────────────────────
        if C > GOOD_THRESH:
            return Action.FORWARD

        if C > WEAK_THRESH:
            return Action.FORWARD

        # ─────────────────────────────────────────────
        # 7. Fallback zone steering
        # ─────────────────────────────────────────────
        if abs(L - R) < 0.05:
            return Action.FORWARD

        self._local_turning = "LEFT" if L > R else "RIGHT"
        return Action.LEFT if self._local_turning == "LEFT" else Action.RIGHT
        
    def _get_path_bias(self):
        """Peeks at the Dijkstra path to figure out the general direction without consuming the edge."""
        snap = self._snap
        if not snap["path"] or snap["node"] is None:
            return None

        try:
            # Find where we are in the path
            idx = snap["path"].index(snap["node"])
            if idx >= len(snap["path"]) - 1:
                return None
            
            src, dst = snap["path"][idx], snap["path"][idx + 1]
            edge = self._graph.get(str(src), {}).get(str(dst), {})
            return edge.get("direction", None) # Expecting "Left", "Right", etc.
        except ValueError:
            # Node isn't in the path (e.g., if Dijkstra hasn't re-planned yet)
            return None
        
    """    
    def _decide(self, L, C, R):
        if C >= GOOD_THRESH:
            action = Action.FORWARD
        elif C >= WEAK_THRESH:
            action = Action.FORWARD
        elif L >= R:
            action = Action.LEFT
        elif R > L:
            action = Action.RIGHT
        else:
            action = Action.BACKWARD
        self._update_heading(action)    # Fix 2: only here and in _global_turn_decide
        return action

    # ── GLOBAL_TURN planner ───────────────────────────────────────────────────

    def _global_turn_decide(self, C, turn_dir):
        Execute commanded turn until center floor clears, then settle.
        if C >= GLOBAL_FWD_THRESH:
            self.fsm_state  = State.GLOBAL_SETTLE
            self._settle_frames = 8
            return Action.FORWARD
        
        self._update_heading(turn_dir)
        return turn_dir
   # ── EVASIVE planner ───────────────────────────────────────────────────────

    def _evasive_decide(self, L, C, R):
        total = L + C + R
        if total >= WEAK_THRESH:
            self._restore_prior_state()
            # Fix 2: _update_heading NOT called here
            return Action.FORWARD
        if L >= R:
            return Action.LEFT
        if R > L:
            return Action.RIGHT
        return Action.BACKWARD
    """
    def _restore_prior_state(self):
        self.fsm_state = self._prior_state

    def _sift_match_count(self, live_img, target_img):
        if live_img is None or target_img is None:
            return 0

        kp_q, des_q = get_cached_features(live_img)
        kp_t, des_t = get_cached_features(target_img)

        if des_q is None or des_t is None:
            return 0

        bf = cv2.BFMatcher()
        raw = bf.knnMatch(des_q, des_t, k=2)

        good = [
            m_n[0] for m_n in raw
            if len(m_n) == 2
            and m_n[0].distance < 0.75 * m_n[1].distance
            and abs(
                kp_q[m_n[0].queryIdx].pt[1]
                - kp_t[m_n[0].trainIdx].pt[1]
            ) <= MAX_Y_DIFF
        ]

        return len(good)

    # ── CONFIRMING -- two-stage CHECKIN (Fix 3) ───────────────────────────────

    def _confirming_act(self):
        """
        Stage 1:
        Confirm that the robot is at the BACK target image.
        If the current FPV matches the back target with enough SIFT matches,
        transition to SEARCH_FRONT.

        Stage 2 happens in _search_front_act().
        """
        if self.fpv is None or self._target_back_img is None:
            return Action.FORWARD

        match_count = self._sift_match_count(self.fpv, self._target_back_img)

        if match_count >= BACK_TARGET_MATCH_THRESH:
            print(
                f"!!! [BACK TARGET CONFIRMED] "
                f"SIFT Matches: {match_count}/{BACK_TARGET_MATCH_THRESH}. "
                f"Transitioning to SEARCH_FRONT. !!!"
            )
            self.fsm_state = State.SEARCH_FRONT
            return Action.RIGHT

        else:
            if self._frame_count % 20 == 0:
                print(
                    f"--- [CONFIRMING BACK] "
                    f"SIFT Matches: {match_count}/{BACK_TARGET_MATCH_THRESH}. "
                    f"Adjusting... ---"
                )

            # Small rotation to better align with the back target
            return Action.RIGHT
            
    def _search_front_act(self):
        """
        After the back target has been confirmed, rotate until the FRONT target
        image is matched. Then fire CHECKIN.
        """
        if self.fpv is None or self._target_front_img is None:
            return Action.RIGHT

        match_count = self._sift_match_count(self.fpv, self._target_front_img)

        if match_count >= FRONT_TARGET_MATCH_THRESH:
            print(
                f"!!! [FRONT TARGET CONFIRMED] "
                f"SIFT Matches: {match_count}/{FRONT_TARGET_MATCH_THRESH}. "
                f"Firing CHECKIN. !!!"
            )
            return Action.CHECKIN

        else:
            if self._frame_count % 20 == 0:
                print(
                    f"--- [SEARCHING FRONT] "
                    f"SIFT Matches: {match_count}/{FRONT_TARGET_MATCH_THRESH}. "
                    f"Rotating... ---"
                )

            return Action.RIGHT

    # ── Heading tracker (Fix 2) ───────────────────────────────────────────────

    def _update_heading(self, action):
        """±90 per intentional LEFT/RIGHT. NOT called from _evasive_decide."""
        if action == Action.LEFT:
            self._heading = (self._heading - 90) % 360
        elif action == Action.RIGHT:
            self._heading = (self._heading + 90) % 360

    # ── UI helpers ────────────────────────────────────────────────────────────

    def show_target_images(self):
        targets = self.get_target_images()
        if not targets:
            return
        while len(targets) < 4:
            targets.append(np.zeros_like(targets[0]))

        hor1       = cv2.hconcat(targets[:2])
        hor2       = cv2.hconcat(targets[2:4])
        concat_img = cv2.vconcat([hor1, hor2])
        h, w       = concat_img.shape[:2]
        col        = (0, 0, 0)
        font, sz, st, ln = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1, cv2.LINE_AA

        concat_img = cv2.line(concat_img, (w//2, 0),  (w//2, h),  col, 2)
        concat_img = cv2.line(concat_img, (0, h//2),  (w, h//2),  col, 2)
        cv2.putText(concat_img, "Front View",  (10, 25),           font, sz, col, st, ln)
        cv2.putText(concat_img, "Left View",   (w//2+10, 25),      font, sz, col, st, ln)
        cv2.putText(concat_img, "Back View",   (10, h//2+25),      font, sz, col, st, ln)
        cv2.putText(concat_img, "Right View",  (w//2+10, h//2+25), font, sz, col, st, ln)

        cv2.imshow("AutoNav: target_images", concat_img)
        cv2.imwrite("target.jpg", concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    # ── Phase-guard state hook ────────────────────────────────────────────────

    def _set_game_state(self, state):
        self._game_state = state


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    import vis_nav_game as vng

    logging.basicConfig(
        filename="auto_nav.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logging.info(f"auto_nav.py using vis_nav_game {vng.core.__version__}")
    vng.play(the_player=AutoNavPlayer())