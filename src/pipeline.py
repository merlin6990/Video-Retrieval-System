from __future__ import annotations
from PIL import Image
import torch
import cv2
import os
import json
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------------------------
# 1. Utility: always give FAISS float32
# ---------------------------------------------------------------------
def _to_float32(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Ensure numpy float32 (required by FAISS).
    Accepts a torch Tensor (any dtype/device) or a numpy array.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


# ---------------------------------------------------------------------
# 2. Video frame extraction
# ---------------------------------------------------------------------
def extract_frames(
    video_path: str,
    frames_per_second: float = 1.0,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Down‑samples a video to `frames_per_second` and returns:
        • list of BGR frames (as numpy arrays)
        • parallel list of metadata dicts  {path: str, time: float}
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / frames_per_second)))

    frames, meta = [], []
    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        meta.append({"path": video_path, "time": frame_idx / fps})

    cap.release()
    return frames, meta


# ---------------------------------------------------------------------
# 3. CLIP → vectors
# ---------------------------------------------------------------------
def convert_frames_to_vectors(
    frames: List[np.ndarray],
    model,
    processor,
) -> np.ndarray:
    """
    Returns a 2‑D numpy float32 array shape (N, D) ready for FAISS.
    """
    pil = [Image.fromarray(f) for f in frames]
    inputs = processor(images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return _to_float32(emb)  # -> np.float32


# ---------------------------------------------------------------------
# 4. Add vectors + save mapping/index
# ---------------------------------------------------------------------
def add_embeddings_with_mapping(
    index: faiss.Index,
    index_path: str,
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    mapping_path: str = "vector_mapping.json",
) -> faiss.Index:

    # Dimension guard
    if vectors.shape[1] != index.d:
        raise ValueError(
            f"Embedding dim {vectors.shape[1]} ≠ index dim {index.d}"
        )

    # Load mapping (keys stored as *strings* consistently)
    mapping: Dict[str, Any]
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    else:
        mapping = {}

    start = index.ntotal
    index.add(vectors)
    for i, md in enumerate(metadata):
        mapping[str(start + i)] = md

    # Persist
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    faiss.write_index(index, index_path)
    return index


# ---------------------------------------------------------------------
# 5. Text query → indices
# ---------------------------------------------------------------------
def retrieve_similar_images_with_text(
    query: str,
    model,
    processor,
    index: faiss.Index,
    top_k: int = 3,
) -> Tuple[str, np.ndarray, np.ndarray]:
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        q_vec = model.get_text_features(**inputs)
    q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=1)
    q_vec = _to_float32(q_vec)
    distances, indices = index.search(q_vec, top_k)
    return query, distances, indices


# ---------------------------------------------------------------------
# 6. Index‑to‑metadata
# ---------------------------------------------------------------------
def get_metadata_from_faiss_indices(
    indices: np.ndarray,
    mapping_path: str = "vector_mapping.json",
) -> List[Any]:
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return [mapping.get(str(idx)) for idx in indices[0]]


# ---------------------------------------------------------------------
# 7. Database helpers (unchanged API)
# ---------------------------------------------------------------------
def initialize_database(
    dimension: int,
    index_path: str,
) -> faiss.Index:
    index = faiss.IndexFlatIP(dimension)
    faiss.write_index(index, index_path)
    return index


def add_to_db(
    video_paths: List[str],
    index: faiss.Index,
    index_path: str,
    model,
    processor,
    mapping_path: str = "vector_mapping.json",
    frames_per_second: float = 1.0,
) -> None:
    for p in video_paths:
        frames, meta = extract_frames(p, frames_per_second)
        vecs = convert_frames_to_vectors(frames, model, processor)
        add_embeddings_with_mapping(
            index, index_path, vecs, meta, mapping_path
        )
    print("Added videos:\n  " + "\n  ".join(video_paths))


def retrieve(
    query: str,
    model,
    processor,
    index: faiss.Index,
    top_k: int = 3,
    mapping_path: str = "vector_mapping.json",
):
    _, _, idxs = retrieve_similar_images_with_text(
        query, model, processor, index, top_k
    )
    return get_metadata_from_faiss_indices(idxs, mapping_path=mapping_path)
