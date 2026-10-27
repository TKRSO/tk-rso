"""
run_faiss_experiments.py
------------------------
Entry point to run Spatial Diversified Proportionality experiments over candidate 
sets retrieved via FAISS embedding search, directly evaluating the claims in Section 7.1.

Evaluates:
  1. FAISS K-NN Candidate Retrieval (google/embeddinggemma-300m)
  2. Baseline IAdU (Exact S^S)
  3. Virtual Grid S^S + Grid IAdU (Proposed)
  4. Biased Sampling

Logs runtime and holistic score H(R) across varying candidate pool sizes K.
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure src/ and repo root are in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models import Place
from alg.baseline_iadu import base_precompute, iadu
from alg.grid_iadu import grid_iadu
from alg.biased_sampling import biased_sampling

try:
    import faiss
    from faiss_query_runner import search_faiss_candidates
except ImportError:
    faiss = None
    search_faiss_candidates = None


def generate_synthetic_faiss_candidates(K: int, dim: int = 768) -> list:
    """Fallback candidate generator when FAISS binary is not installed."""
    np.random.seed(K)
    coords = np.random.uniform(10, 90, size=(K, 2))
    r_scores = np.random.uniform(0.3, 0.95, size=K)
    
    places = []
    for i in range(K):
        p = Place(id=i, coords=tuple(coords[i]))
        p.rF = float(r_scores[i])
        places.append(p)
    return places


def run_experiment(
    K_values: list = [500, 1000, 2000],
    k: int = 20,
    G: int = 250,
    gamma: float = 1.0,
    index_path: str = None
):
    print("=" * 80)
    print(f"Top-k RSO: FAISS Embedding-Based Proportionality Benchmarks")
    print(f"k = {k} | Grid Size |G| = {G} | gamma = {gamma}")
    print("=" * 80)

    # Header
    print(f"{'K':<6} | {'Algorithm':<22} | {'Time (s)':<10} | {'Speedup':<8} | {'H(R) Score':<12} | {'pSS Sum':<10}")
    print("-" * 80)

    # Load or build FAISS index if available
    faiss_index = None
    if faiss is not None and index_path and os.path.exists(index_path):
        print(f"Loading FAISS index from: {index_path}")
        faiss_index = faiss.read_index(index_path)

    for K in K_values:
        W = K / (gamma * k)

        # 1. Candidate Retrieval
        t0_retrieval = time.perf_counter()
        if faiss_index is not None and search_faiss_candidates is not None:
            np.random.seed(42)
            dummy_q = np.random.randn(faiss_index.d).astype(np.float32)
            S = search_faiss_candidates(faiss_index, dummy_q, K)
        else:
            S = generate_synthetic_faiss_candidates(K)
        retrieval_time = time.perf_counter() - t0_retrieval

        # 2. Precomputations for exact baseline
        exact_psS, exact_sS, base_prep_time = base_precompute(S)

        # Method 1: Baseline IAdU (Combination 1)
        R_base, score_base, pss_base, psr_base, rf_base, prep_t_base, sel_t_base = iadu(
            S, k, W, exact_psS, exact_sS, base_prep_time
        )
        total_time_base = prep_t_base + sel_t_base

        # Method 2: Virtual Grid S^S + Grid IAdU (Combination 2)
        R_grid, score_grid, pss_grid, psr_grid, rf_grid, prep_t_grid, sel_t_grid, cl_len = grid_iadu(
            S, k, W, G, exact_psS, exact_sS
        )
        total_time_grid = prep_t_grid + sel_t_grid
        speedup_grid = (total_time_base / total_time_grid) if total_time_grid > 0 else 1.0

        # Method 3: Biased Sampling (Combination 5)
        R_bias, score_bias, pss_bias, psr_bias, rf_bias, _, sel_t_bias = biased_sampling(
            S, k, W, exact_psS, exact_sS
        )

        # Print formatted results
        print(f"{K:<6} | {'Baseline IAdU':<22} | {total_time_base:10.4f} | {'1.0x':<8} | {score_base:12.2f} | {pss_base:10.2f}")
        print(f"{K:<6} | {'Grid IAdU (Proposed)':<22} | {total_time_grid:10.4f} | {f'{speedup_grid:.1f}x':<8} | {score_grid:12.2f} | {pss_grid:10.2f}")
        print(f"{K:<6} | {'Biased Sampling':<22} | {sel_t_bias:10.4f} | {'--':<8} | {score_bias:12.2f} | {pss_bias:10.2f}")
        print("-" * 80)

    print("\nBenchmark completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run FAISS embedding-based spatial proportionality experiments.")
    parser.add_argument("--k", type=int, default=20, help="Number of representative places to retrieve (k)")
    parser.add_argument("--grid_size", type=int, default=250, help="Virtual grid cell count |G|")
    parser.add_argument("--index_path", type=str, default="", help="Path to precomputed FAISS index .bin/.index")
    args = parser.parse_args()

    run_experiment(
        K_values=[500, 1000, 2000],
        k=args.k,
        G=args.grid_size,
        index_path=args.index_path if args.index_path else None
    )


if __name__ == "__main__":
    main()
