"""
build_graph.py
==============
Offline pipeline functions imported by player.py.

Contains everything that touches raw frames, descriptors, and graph construction:
    extract_keyframes_uniform
    detect_turns                   (batch / offline version)
    detect_turns_single_frame      (per-frame, called from player.py see())
    merge_and_sort
    deduplicate / deduplicate_keyframes
    DINOv2Descriptor               (class)
    encode_geometric
    build_descriptor_index / build_faiss_index
    infer_action_from_flow
    infer_action
    build_pose_graph
    load_graph
    save_artifacts

Fixes applied vs original:
    [1] extract_keyframes_uniform  — frames now sorted numerically before sampling
    [2] detect_turns               — frames now sorted numerically before flow
    [3] deduplicate                — corrupt/unreadable frames skipped, not kept
    [4] build_pose_graph           — loop closures added in BOTH directions
    [5] build_pose_graph           — sequential edge progress bar added
    [6] save_artifacts             — similarity=None replaced with 0.0 in JSON
    [7] load_graph                 — now imported and used by player.py; kept here
    [8] build_pose_graph           — descriptors forced to contiguous float32
                                     before FAISS batch search; search done in
                                     row-by-row chunks to prevent segfault on
                                     Apple Silicon (MPS / ARM)
"""

import json
import os
import pickle
from pathlib import Path

import cv2
import networkx as nx
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. KEYFRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _sorted_frames(frame_dir: str) -> list[Path]:
    """
    Return all image files in frame_dir sorted by numeric stem, falling back
    to lexicographic order when stems are not purely numeric.

    FIX [1,2]: Path.iterdir() returns files in arbitrary filesystem order.
    Sorting ensures frame_idx values are meaningful and optical-flow pairs
    are temporally adjacent.
    """
    frames = [
        f for f in Path(frame_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    frames.sort(key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
    return frames


def extract_keyframes_uniform(frame_dir: str, step: int = 40) -> list[dict]:
    """
    Return every `step`-th frame as a candidate keyframe dict.

    Offline / notebook usage:
        uniform_kfs = extract_keyframes_uniform("/path/to/frames", step=40)

    Live usage: player.py replicates this logic inline inside see() so it
    runs frame-by-frame without loading the whole directory.
    """
    # FIX [1]: use sorted helper so frame_idx reflects true temporal position
    frames = _sorted_frames(frame_dir)

    keyframes = []
    for i, fpath in enumerate(frames):
        if i % step == 0:
            keyframes.append({
                "frame_idx": i,
                "path": str(fpath),
                "source": "uniform",
            })

    print(f"[uniform]  {len(keyframes)} candidates from {len(frames)} frames "
          f"(step={step})")
    return keyframes


def detect_turns(frame_dir: str,
                 rotation_threshold: float = 15.0,
                 min_gap: int = 10) -> list[dict]:
    """
    Batch / offline turn detection. Reads an entire saved frame directory and
    returns a list of turning-point keyframe dicts.

    Offline / notebook usage:
        turn_kfs = detect_turns("/path/to/frames", rotation_threshold=15.0)

    Live usage: player.py calls detect_turns_single_frame() per frame instead.
    min_gap enforces a minimum gap between detected turns (avoids clustering).
    """
    # FIX [2]: sort frames so optical-flow pairs are temporally adjacent
    frames = _sorted_frames(frame_dir)

    turn_frames, prev_gray = [], None
    last_turn = -min_gap  # Allow detection at the very start of the sequence

    for i, fpath in enumerate(frames):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and (i - last_turn) > min_gap:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            fx = flow[..., 0]
            w  = gray.shape[1]
            rotation_signal = abs(
                fx[:, :w // 2].mean() - fx[:, w // 2:].mean()
            )

            if rotation_signal > rotation_threshold:
                turn_frames.append({
                    "frame_idx": i,
                    "path": str(fpath),
                    "source": "turn",
                    "rotation_signal": float(rotation_signal),
                })
                last_turn = i

        prev_gray = gray

    print(f"[turns]    {len(turn_frames)} turning-point candidates detected")
    return turn_frames


def detect_turns_single_frame(prev_gray: np.ndarray,
                               gray: np.ndarray,
                               frame_idx: int,
                               fpv: np.ndarray,
                               threshold: float,
                               last_turn: int,
                               min_gap: int = 10,
                               save_dir: str | None = None) -> dict | None:
    """
    Per-frame turn detection called inside player.py see() every frame.
    Returns a keyframe dict if a turn is detected, else None.
    Writes the frame image to save_dir so the keyframe path is valid later.
    """
    if (frame_idx - last_turn) <= min_gap:
        return None

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    fx = flow[..., 0]
    w  = gray.shape[1]
    rotation_signal = abs(fx[:, :w // 2].mean() - fx[:, w // 2:].mean())

    if rotation_signal > threshold:
        if save_dir is None:
            import tempfile
            save_dir = os.path.join(tempfile.gettempdir(), "vis_nav_kf_frames")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"turn_{frame_idx:06d}.jpg")
        cv2.imwrite(path, fpv)
        return {
            "frame_idx": frame_idx,
            "path": path,
            "source": "turn",
            "rotation_signal": float(rotation_signal),
        }
    return None


def merge_and_sort(candidates: list[dict]) -> list[dict]:
    """
    Merge uniform + turn candidate lists, deduplicate by frame index, sort.
    When both sources produce a keyframe at the same index, 'turn' wins.
    """
    seen = {}
    for kf in candidates:
        fi = kf["frame_idx"]
        if fi not in seen or kf.get("source") == "turn":
            seen[fi] = kf
    merged = sorted(seen.values(), key=lambda x: x["frame_idx"])
    print(f"[merge]    {len(merged)} candidates after merge")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(keyframes: list[dict], hamming_threshold: int = 8) -> list[dict]:
    """
    Remove near-duplicate keyframes using pHash (imagehash library).
    Gracefully skips deduplication if imagehash is not installed.

    FIX [3]: Unreadable / corrupt frames are now SKIPPED rather than blindly
    kept — keeping them without a hash meant they were never deduplicated
    against anything, inflating the keyframe set with bad data.

    Called once in pre_navigation() after merge_and_sort().
    Exported as both `deduplicate` and `deduplicate_keyframes`.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("[dedup]    imagehash not available — skipping deduplication")
        return keyframes

    kept, hashes = [], []

    for kf in keyframes:
        try:
            h = imagehash.phash(Image.open(kf["path"]))
        except Exception as e:
            # FIX [3]: skip unreadable frames instead of keeping them blindly
            print(f"[dedup]    Skipping unreadable frame: {kf['path']} ({e})")
            continue

        if all(h - ex > hamming_threshold for ex in hashes):
            kept.append(kf)
            hashes.append(h)

    print(f"[dedup]    {len(keyframes) - len(kept)} duplicates removed "
          f"→ {len(kept)} keyframes")
    return kept


# Alias — player.py and the notebook may import either name
deduplicate_keyframes = deduplicate


# ─────────────────────────────────────────────────────────────────────────────
# 3. DINOV2 DESCRIPTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Descriptor:
    """
    Wraps a DINOv2 ViT model for scene-level descriptor extraction.

    The CLS token encodes structural / geometric scene context rather than
    surface texture, making it robust to the 200-texture generalisation
    constraint.

    Lifecycle in player.py:
        __init__        — model is loaded ONCE here (~4 s, paid at startup)
        pre_navigation  — encode_geometric() batch-encodes all keyframes
        act()           — localize() / localize_robust() queries per frame
    """

    def __init__(self, model_size: str = "s", device: str | None = None):
        import torch
        import torchvision.transforms as T

        # Determine device: prioritize CUDA, then MPS (Apple Silicon), fallback to CPU
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        print(f"[DINOv2]   Loading ViT-{model_size.upper()}/14 on {self.device}…")
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            f"dinov2_vit{model_size}14",
            pretrained=True,
        ).to(self.device).eval()

        # Preprocessing pipeline: resize, crop, normalize for ImageNet standards
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        print(f"[DINOv2]   Ready ({self.device})")

    def encode(self, img_path: str) -> np.ndarray:
        """
        Encode a single image file.
        Returns an L2-normalised float32 vector (shape: [D]).
        Called by localize() and localize_robust() at navigation time.
        """
        import torch
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        x   = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_batch(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode many image files efficiently.
        Returns float32 array of shape (N, D).
        Called by encode_geometric() in pre_navigation().
        """
        import torch
        from PIL import Image

        all_feats, total = [], len(paths)
        for start in range(0, total, batch_size):
            batch = paths[start:start + batch_size]
            imgs  = torch.stack([
                self.transform(Image.open(p).convert("RGB")) for p in batch
            ]).to(self.device)
            with torch.no_grad():
                feats = self.model(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
            done = min(start + batch_size, total)
            print(f"\r[DINOv2]   Encoded {done}/{total} "
                  f"({done * 100 // total}%)", end="", flush=True)
        print()
        return np.vstack(all_feats).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE_GEOMETRIC
# ─────────────────────────────────────────────────────────────────────────────

def encode_geometric(encoder: DINOv2Descriptor,
                     paths: list[str],
                     batch_size: int = 32) -> np.ndarray:
    """
    Geometry-biased descriptor encoding.

    Uses patch-token mean pooling rather than the raw CLS token.
    Patch tokens respond more strongly to structural layout (wall edges,
    corridor junctions, openings) and less to surface texture — which helps
    with the 200-texture generalisation constraint.

    Falls back to standard CLS encoding if DINOv2's forward_features()
    is unavailable (older torch.hub builds).

    Called once in pre_navigation() to encode all keyframes.
    """
    import torch
    from PIL import Image

    all_feats, total = [], len(paths)
    for start in range(0, total, batch_size):
        batch = paths[start:start + batch_size]
        imgs  = torch.stack([
            encoder.transform(Image.open(p).convert("RGB")) for p in batch
        ]).to(encoder.device)
        with torch.no_grad():
            try:
                out          = encoder.model.forward_features(imgs)
                patch_tokens = out["x_norm_patchtokens"]   # (B, N_patches, D)
                feats        = patch_tokens.mean(dim=1)    # Average over patches
            except (AttributeError, KeyError):
                # Fallback: CLS token
                feats = encoder.model(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        done = min(start + batch_size, total)
        print(f"\r[geometric] {done}/{total}", end="", flush=True)
    print()
    return np.vstack(all_feats).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FAISS INDEX
# ─────────────────────────────────────────────────────────────────────────────

def build_descriptor_index(descriptors: np.ndarray,
                            keyframes: list[dict] = None):
    """
    Build an IndexFlatIP (inner product = cosine similarity after L2 norm).
    At query time a single index.search(q, k) call returns top-k matches
    in ~20 ms regardless of index size.

    Called once in pre_navigation() after encode_geometric().
    Exported as both `build_descriptor_index` and `build_faiss_index`.
    """
    import faiss

    D     = descriptors.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(descriptors)
    print(f"[faiss]    Index: {index.ntotal} vectors, dim={D}")
    return index


# Alias
build_faiss_index = build_descriptor_index


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACTION INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def infer_action_from_flow(prev_gray: np.ndarray, gray: np.ndarray) -> str:
    """
    Classify dominant motion between two grayscale frames via optical flow.
    Returns: 'forward' | 'turn_left' | 'turn_right' | 'backward'

    Called in two places:
        player.py see()     — every exploration frame → builds command_log
        build_pose_graph()  — when infer_actions=True to label sequential edges
    """
    flow    = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    fx, fy  = flow[..., 0], flow[..., 1]
    h, w    = prev_gray.shape
    lateral = fx[:, 2 * w // 3:].mean() - fx[:, :w // 3].mean()
    forward = -fy[:, w // 3: 2 * w // 3].mean()

    if abs(lateral) > 3.0:
        return "turn_right" if lateral > 0 else "turn_left"
    elif forward > 1.5:
        return "forward"
    elif forward < -1.5:
        return "backward"
    return "forward"


def infer_action(img1_path: str, img2_path: str) -> str:
    """
    Path-based wrapper around infer_action_from_flow for offline / notebook use.
    Called by build_pose_graph() when labelling sequential edges.
    """
    g1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)
    if g1 is None or g2 is None:
        return "forward"
    return infer_action_from_flow(g1, g2)


# ─────────────────────────────────────────────────────────────────────────────
# 7. POSE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_pose_graph(
    keyframes: list[dict],
    descriptors: np.ndarray,
    index,
    loop_threshold: float = 0.85,
    loop_min_gap: int = 50,
    infer_actions: bool = True,
) -> nx.DiGraph:
    """
    Build a directed pose graph from keyframes + descriptors.

    Nodes carry: frame_idx, path, source.
    Edge types:
        sequential   KF[i] → KF[i+1]  weight=1.0  action from optical flow
        loop_closure KF[i] → KF[j]    weight=0.5  when cosine_sim > threshold
                     KF[j] → KF[i]                and frame gap > loop_min_gap
                                                   (FIX [4]: BOTH directions)

    FIX [4]: Original code only added loop closure edges in one direction
             (i → j). This halved the usable shortcuts for route planning.
             Both i→j and j→i are now added.

    FIX [5]: Sequential edge progress bar added — without it the function
             silently hangs for minutes on large datasets.

    FIX [8]: FAISS batch search replaced with a row-by-row loop on a
             contiguous float32 copy of descriptors.  The original single
             index.search(descriptors, 15) call segfaulted on Apple Silicon
             (MPS/ARM) because the descriptor array returned by
             encode_geometric() may not be C-contiguous after vstack, and
             FAISS's ARM SIMD path reads past buffer boundaries when the
             memory layout is unexpected.  Searching one row at a time
             eliminates the large contiguous read and is safe on all
             platforms.  Runtime cost is negligible (~1 ms per row on CPU).
    """
    G = nx.DiGraph()
    N = len(keyframes)

    # Add all keyframe nodes
    for i, kf in enumerate(keyframes):
        G.add_node(i, **{k: v for k, v in kf.items()})

    # ── Sequential edges ──────────────────────────────────────────────────────
    print("[graph]    Adding sequential edges…")
    for i in range(N - 1):
        action = "forward"
        if infer_actions:
            try:
                action = infer_action(
                    keyframes[i]["path"], keyframes[i + 1]["path"]
                )
            except Exception:
                pass
        G.add_edge(i, i + 1, edge_type="sequential", action=action, weight=1.0)

        # FIX [5]: progress bar for sequential edges (can be slow on large sets)
        if (i + 1) % 100 == 0 or i == N - 2:
            print(f"\r[graph]    Sequential edges: {i + 1}/{N - 1}",
                  end="", flush=True)
    print()

    # ── Loop closure edges ────────────────────────────────────────────────────
    print("[graph]    Searching for loop closures…")
    lc_count = 0

    # FIX [8]: force a fresh contiguous float32 array before any FAISS call.
    # np.vstack output from encode_geometric is nominally float32 but may not
    # be C-contiguous on all platforms, which triggers a segfault in FAISS's
    # ARM SIMD path on Apple Silicon.
    descriptors_cpu = np.ascontiguousarray(descriptors, dtype=np.float32)

    # FIX [8]: search row-by-row instead of passing the full matrix at once.
    # A single index.search(descriptors, 15) with N=1000+ rows segfaults on
    # MPS/ARM.  One row at a time is safe on every platform and fast enough
    # (~1 ms/row on CPU FAISS).
    all_scores = []
    all_idxs   = []
    for i in range(N):
        row = descriptors_cpu[i : i + 1]          # shape (1, D) — always contiguous
        s, idx = index.search(row, 15)
        all_scores.append(s[0])                   # 1-D array of 15 scores
        all_idxs.append(idx[0])                   # 1-D array of 15 indices
        if (i + 1) % 100 == 0 or i == N - 1:
            print(f"\r[graph]    Loop closure search: {i + 1}/{N}",
                  end="", flush=True)
    print()

    for i, (scores, idxs) in enumerate(zip(all_scores, all_idxs)):
        for score, j in zip(scores, idxs):
            if j == i or j < 0:
                continue
            gap = abs(
                keyframes[i]["frame_idx"] - keyframes[j]["frame_idx"]
            )
            if gap > loop_min_gap and float(score) > loop_threshold:
                lc_attrs = dict(
                    edge_type="loop_closure",
                    action="loop",
                    weight=0.5,
                    similarity=float(score),
                )
                # FIX [4]: add loop closure in BOTH directions so the
                # planner can traverse shortcuts regardless of travel direction
                if not G.has_edge(i, j):
                    G.add_edge(i, j, **lc_attrs)
                    lc_count += 1
                if not G.has_edge(j, i):
                    G.add_edge(j, i, **lc_attrs)
                    lc_count += 1

    print(f"[graph]    {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges ({lc_count} loop-closure half-edges)")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(path: str = "maze_graph.pkl") -> nx.DiGraph | None:
    """
    Load a previously saved networkx DiGraph from disk.

    player.py imports and calls this in _load_existing_map() so the load
    logic is not duplicated between build_graph.py and player.py.

    Returns None if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        graph = pickle.load(f)
    print(f"[load_graph]  Loaded {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges from {path}")
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalar types to native Python before JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def save_artifacts(output_dir: str,
                   keyframes: list[dict],
                   descriptors: np.ndarray,
                   index,
                   graph: nx.DiGraph) -> dict:
    """
    Persist all pipeline artefacts to disk.

    Writes:
        keyframe_index.faiss   — FAISS index for runtime lookup
        keyframe_meta.pkl      — keyframes list + descriptor matrix
        maze_graph.pkl         — full networkx DiGraph
        graph_export.json      — lightweight JSON for HTML visualiser

    FIX [6]: similarity field in JSON export now defaults to 0.0 instead of
             None, preventing downstream null-dereference errors when
             comparing similarity scores.

    Called once at the end of pre_navigation().
    """
    import faiss

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = str(out / "keyframe_index.faiss")
    meta_path  = str(out / "keyframe_meta.pkl")
    graph_path = str(out / "maze_graph.pkl")
    json_path  = str(out / "graph_export.json")

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump({"keyframes": keyframes, "descriptors": descriptors}, f)

    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    nodes_json = [
        {
            "id":        int(nid),
            "frame_idx": int(d.get("frame_idx", nid)),
            "source":    str(d.get("source", "uniform")),
            "path":      os.path.basename(str(d.get("path", ""))),
        }
        for nid, d in graph.nodes(data=True)
    ]
    edges_json = [
        {
            "source":    int(u),
            "target":    int(v),
            "edge_type": str(d.get("edge_type", "sequential")),
            "action":    str(d.get("action", "forward")),
            "weight":    float(d.get("weight", 1.0)),
            # FIX [6]: default to 0.0 instead of None to avoid downstream
            #          null-dereference when consumers compare similarity values
            "similarity": float(d["similarity"]) if d.get("similarity") is not None else 0.0,
        }
        for u, v, d in graph.edges(data=True)
    ]

    with open(json_path, "w") as f:
        json.dump({"nodes": nodes_json, "edges": edges_json},
                  f, cls=_NumpyEncoder)

    paths = {
        "faiss_index": index_path,
        "metadata":    meta_path,
        "graph":       graph_path,
        "json_export": json_path,
    }
    print("\n[save]     Artefacts written:")
    for k, v in paths.items():
        size = os.path.getsize(v) / 1024
        print(f"           {k:20s} → {v}  ({size:.1f} KB)")
    return paths