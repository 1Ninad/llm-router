import torch
import json
import numpy as np
from openai import OpenAI
from matrix_factorization import MatrixFactorizationRouter
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print("Using device:", DEVICE)

# load trained router
checkpoint = torch.load("router_sw.pt", map_location=DEVICE)

dq = checkpoint["model_params"]["dq"]
dm = checkpoint["model_params"]["dm"]
num_models = checkpoint["model_params"]["num_models"]
router = MatrixFactorizationRouter(num_models, dq, dm).to(DEVICE)
router.load_state_dict(checkpoint["state_dict"])
router.eval()

# load model index
model_index = json.load(open("train_model_index.json"))


def embed_query(query):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(emb.data[0].embedding, dtype=np.float32)


def route(query, strong_model, weak_model):
    q = embed_query(query)
    q = torch.tensor(q).unsqueeze(0).to(DEVICE)

    strong_id = torch.tensor([model_index[strong_model]]).to(DEVICE)
    weak_id = torch.tensor([model_index[weak_model]]).to(DEVICE)

    with torch.no_grad():
        p = router.win_probability(strong_id, weak_id, q)

    p = p.item()

    if p > 0.5:
        chosen = strong_model
    else:
        chosen = weak_model

    return {
        "prob_strong_wins": p,
        "chosen_model": chosen
    }