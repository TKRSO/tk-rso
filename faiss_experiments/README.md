# FAISS Embedding-Based Spatial Proportionality Experiments

This directory contains the indexing and candidate retrieval pipeline using **FAISS** [27] and **Google EmbeddingGemma-300M** [16], as described in Section 7.1 of the paper:

> *"In the second type, we use embeddings, where $r(p_i)$ represents the embedding similarity between the query and $p_i$. We embedded our objects using the google/embedding-gemma-300m [16] model and indexed them in FAISS [27]."*

---

## Architecture Overview

```text
Natural-Language / Entity Query
               │
               ▼
   [google/embeddinggemma-300m]  ──> 768-d Query Vector
               │
               ▼
     [FAISS L2 Vector Index]     ──> Fast retrieval of Top-K candidates (K = 500..5000)
               │                     with embedding relevance scores r(p_i)
               ▼
 [Top-k RSO Framework (src/alg/)] ──> Evaluates Baseline IAdU vs. Virtual Grid IAdU
                                      to select k = 20 places
```

---

## Directory Contents

| File | Description |
| :--- | :--- |
| `generate_place_embeddings.py` | Encodes place entity descriptions into 768-d vectors and builds a FAISS `IndexFlatL2` or `IndexIVFFlat` index. Supports `--sample` for testing without downloading full datasets. |
| `faiss_query_runner.py` | Queries the FAISS index to retrieve the closest $K$ candidates and transforms $L_2$ distances into normalized relevance scores $r(p_i) \in [0, 1]$. |
| `run_faiss_experiments.py` | Benchmark runner executing the proportionality selection over the FAISS candidate pool across $K \in \{500, 1000, 2000\}$. |
| `requirements.txt` | Python packages required for indexing. |

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Experiment
```bash
python run_faiss_experiments.py --k 20 --grid_size 250
```

### 3. Generate a Sample FAISS Index
```bash
python generate_place_embeddings.py --sample --output_dir ./faiss_index
```
