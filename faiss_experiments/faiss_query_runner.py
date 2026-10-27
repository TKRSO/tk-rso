"""
faiss_query_runner.py
----------------------
Executes candidate retrieval using FAISS exact / approximate nearest neighbor search
and formats retrieved candidates into Place objects with normalized relevance scores r(p_i).

Supports both pre-computed query vectors and text queries via SentenceTransformers.
"""

import os
import sys
import numpy as np
from typing import List, Tuple

# Add parent directory and src to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PARENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models import Place

try:
    import faiss
except ImportError:
    faiss = None


def search_faiss_candidates(
    index,
    query_vector: np.ndarray,
    K: int,
    place_coordinates: np.ndarray = None,
    place_ids: np.ndarray = None
) -> List[Place]:
    """
    Retrieves the top-K nearest neighbors from the FAISS index for a query vector,
    converts L2 distances into normalized relevance scores r(p_i), and returns Place objects.
    """
    if faiss is None:
        raise ImportError("FAISS is not installed.")

    query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
    distances, indices = index.search(query_vector, K)

    dists = distances[0]
    idx_list = indices[0]

    # Convert L2 distance into relevance score r(p_i) in [0, 1]
    # Smaller distance -> Higher relevance
    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
    min_dist = np.min(dists)
    range_dist = max_dist - min_dist if max_dist > min_dist else 1.0
    relevance_scores = 1.0 - ((dists - min_dist) / range_dist)

    places = []
    for rank, (idx, r_score) in enumerate(zip(idx_list, relevance_scores)):
        pid = int(place_ids[idx]) if place_ids is not None else int(idx)
        
        # Assign spatial coordinates (x, y)
        if place_coordinates is not None and idx < len(place_coordinates):
            coords = tuple(place_coordinates[idx][:2])
        else:
            # Deterministic pseudo-coordinates for indexing evaluation benchmarks
            np.random.seed(pid)
            coords = (float(np.random.uniform(0, 100)), float(np.random.uniform(0, 100)))

        place = Place(id=pid, coords=coords)
        place.rF = float(r_score)
        places.append(place)

    return places


def run_benchmark_faiss_search(
    index_path: str,
    K_values: List[int] = [500, 1000, 2000, 5000],
    num_queries: int = 5,
    dim: int = 768
) -> dict:
    """
    Runs batched FAISS search benchmarks across candidate pool sizes K.
    """
    if faiss is None:
        raise ImportError("FAISS is not installed.")

    if not os.path.exists(index_path):
        print(f"Index file {index_path} not found. Building dummy in-memory FlatL2 index...")
        index = faiss.IndexFlatL2(dim)
        dummy_data = np.random.randn(max(K_values) * 2, dim).astype(np.float32)
        index.add(dummy_data)
    else:
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)

    np.random.seed(42)
    query_vectors = np.random.randn(num_queries, dim).astype(np.float32)

    import time
    timings = {}
    for K in K_values:
        t0 = time.perf_counter()
        distances, indices = index.search(query_vectors, K)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / num_queries
        timings[K] = elapsed_ms
        print(f"  K = {K:5d} | Avg FAISS Retrieval Latency: {elapsed_ms:.3f} ms")

    return timings


if __name__ == "__main__":
    print("Testing FAISS query runner benchmark...")
    if faiss is not None:
        run_benchmark_faiss_search("dummy.index", K_values=[100, 500, 1000])
    else:
        print("FAISS is not installed. Please install faiss-cpu to execute.")
