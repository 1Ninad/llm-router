# train_router.py
# Matrix-Factorization router training (strong-vs-weak).
# Hyperparameters: dm=128, batch_size=64, epochs=10, Adam(lr=3e-4, weight_decay=1e-5)

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



DM = 128
BATCH_SIZE = 64        
EPOCHS = 10            
LR = 3e-4              
WEIGHT_DECAY = 1e-5    


TRAIN_PQ = "data/train_strong_weak.parquet"   
EMB_FILE = "train_embeddings.npy"        
MODEL_INDEX_FILE = "train_model_index.json"
OUT_MODEL = "router_sw.pt"


if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("Using device:", DEVICE)


try:
    torch.set_num_threads(4)
except Exception:
    pass

class RouterDataset(Dataset):
    def __init__(self, train_pq=TRAIN_PQ, emb_file=EMB_FILE):
        df = pd.read_parquet(train_pq)
        emb = np.load(emb_file)
        assert len(df) == len(emb), f"train rows {len(df)} != embeddings {len(emb)}"
        self.df = df.reset_index(drop=True)
        self.q_emb = emb.astype("float32")
        # build model index for unique models present in training (strong+weak)
        models = pd.unique(self.df[["strong_model","weak_model"]].values.ravel())
        self.model_to_id = {m:i for i,m in enumerate(models)}
        json.dump(self.model_to_id, open(MODEL_INDEX_FILE, "w"), indent=2)
        self.strong_ids = np.array([self.model_to_id[m] for m in self.df["strong_model"].tolist()], dtype=np.int64)
        self.weak_ids = np.array([self.model_to_id[m] for m in self.df["weak_model"].tolist()], dtype=np.int64)
        self.labels = np.array(self.df["strong_won"].tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return (q_emb, strong_id, weak_id, label)
        return (torch.from_numpy(self.q_emb[idx]), 
                torch.tensor(self.strong_ids[idx], dtype=torch.long),
                torch.tensor(self.weak_ids[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32))



def train_loop(device=DEVICE, batch_size=BATCH_SIZE):
    ds = RouterDataset()
    dq = ds.q_emb.shape[1]
    num_models = len(ds.model_to_id)
    print(f"Training rows: {len(ds)}, dq={dq}, num_models={num_models}")

    # DataLoader settings: pin_memory is helpful on CUDA; harmless on CPU; for MPS it's ignored.
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = MatrixFactorizationRouter(num_models, dq, DM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i, (q_emb, s_ids, w_ids, labels) in enumerate(dl):
            # move to device
            q_emb = q_emb.to(device)
            s_ids = s_ids.to(device)
            w_ids = w_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            probs = model.win_probability(s_ids, w_ids, q_emb)
            loss = loss_fn(probs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * q_emb.size(0)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.6f}")

    # Save checkpoint including metadata
    torch.save({"state_dict": model.state_dict(), "model_params": {"dq": dq, "dm": DM, "num_models": num_models}}, OUT_MODEL)
    print("Saved", OUT_MODEL)
    print("Model mapping saved to", MODEL_INDEX_FILE)


if __name__ == "__main__":
    try:
        train_loop(DEVICE, BATCH_SIZE)
    except RuntimeError as e:
        print("RuntimeError during training:", str(e))
        if DEVICE == "mps" and BATCH_SIZE > 16:
            new_bs = max(16, BATCH_SIZE // 2)
            print(f"Retrying with smaller BATCH_SIZE={new_bs}")
            train_loop(DEVICE, new_bs)
        else:
            raise