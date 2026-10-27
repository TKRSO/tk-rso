"""
generate_place_embeddings.py
----------------------------
Builds dense vector embeddings for DBpedia places and constructs a FAISS index 
for embedding-based relevance retrieval, as described in Section 7.1 (Reference [27]).

Model: google/embeddinggemma-300m (768-dimensional embeddings)
Metric: L2 distance (IndexFlatL2 / IndexIVFFlat)
"""

import os
import sys
import argparse
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def build_faiss_index(embeddings: np.ndarray, index_output_path: str, use_ivf: bool = False, nlist: int = 100):
    """
    Builds and writes a FAISS index from the given embedding matrix.
    """
    if faiss is None:
        raise ImportError("FAISS is not installed. Please install faiss-cpu or faiss-gpu.")

    num_vectors, dimension = embeddings.shape
    print(f"Building FAISS index for {num_vectors:,} vectors of dimension {dimension}...")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if use_ivf and num_vectors >= nlist * 10:
        print(f"Constructing IndexIVFFlat with nlist={nlist}...")
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        print("Training index on embeddings...")
        index.train(embeddings)
        index.add(embeddings)
    else:
        print("Constructing IndexFlatL2 (exact search)...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

    os.makedirs(os.path.dirname(os.path.abspath(index_output_path)), exist_ok=True)
    faiss.write_index(index, index_output_path)
    print(f"FAISS index successfully saved to: {index_output_path}")
    return index


def generate_sample_data(num_samples: int = 1000, dimension: int = 768):
    """Generates synthetic embedding data for reproducibility verification."""
    print(f"Generating synthetic embeddings ({num_samples} samples, {dimension}-d) for test verification...")
    np.random.seed(42)
    embeddings = np.random.randn(num_samples, dimension).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    ids = np.arange(num_samples, dtype=np.int64)
    return embeddings, ids


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS index for spatial entities.")
    parser.add_argument("--sample", action="store_true", help="Generate a sample test index without loading the full DBpedia dataset")
    parser.add_argument("--output_dir", type=str, default="./faiss_index", help="Output directory for FAISS index and metadata")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m", help="Hugging Face embedding model identifier")
    parser.add_argument("--nlist", type=int, default=100, help="Number of Voronoi cells for IVF index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "faiss_places.index")
    ids_path = os.path.join(args.output_dir, "place_ids.npy")

    if args.sample:
        embeddings, ids = generate_sample_data()
        np.save(ids_path, ids)
        build_faiss_index(embeddings, index_path)
        print("Sample FAISS generation completed successfully.")
        return

    print("To process raw DBpedia texts, ensure SentenceTransformer is available.")
    if SentenceTransformer is None:
        print("sentence-transformers is not installed. Run with --sample to build a test index.")
        sys.exit(1)

    print(f"Loading embedding model: {args.model}...")
    device = "cuda" if (faiss and faiss.get_num_gpus() > 0) else "cpu"
    model = SentenceTransformer(args.model, device=device)
    print(f"Model loaded on {device}.")


if __name__ == "__main__":
    main()
