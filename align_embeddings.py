# align_embeddings_to_train.py
# Produces train_embeddings.npy aligned with train_strong_weak.parquet
import numpy as np
import pandas as pd

ORIG_PQ = "data/dataset_chatbot_arena.parquet"
EMB_FILE = "query_embeddings.npy"
TRAIN_PQ = "data/train_strong_weak.parquet"
OUT_EMB = "train_embeddings.npy"

def main():
    orig = pd.read_parquet(ORIG_PQ)
    train = pd.read_parquet(TRAIN_PQ)

    emb = np.load(EMB_FILE, allow_pickle=False)
    assert len(orig) == len(emb), f"orig rows {len(orig)} != embeddings {len(emb)}"

    # Build map: query_text -> first index
    query_to_idx = {}
    for i, q in enumerate(orig["query"].astype(str).tolist()):
        if q not in query_to_idx:
            query_to_idx[q] = i

    train_embeddings = []
    missing = 0
    for q in train["query"].astype(str).tolist():
        idx = query_to_idx.get(q, None)
        if idx is None:
            # fallback: try stripping whitespace
            idx = query_to_idx.get(q.strip(), None)
        if idx is None:
            missing += 1
            train_embeddings.append(np.zeros_like(emb[0]))  # placeholder; better fix missing cases manually
        else:
            train_embeddings.append(emb[idx])

    train_embeddings = np.stack(train_embeddings, axis=0)
    np.save(OUT_EMB, train_embeddings)
    print("WROTE", OUT_EMB, "shape=", train_embeddings.shape, "missing matches=", missing)

if __name__ == "__main__":
    main()