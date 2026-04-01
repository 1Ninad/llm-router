import argparse
import pandas as pd
import torch
import json
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matrix_factorization import MatrixFactorizationRouter

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
or_key = os.getenv("OPENROUTER_API_KEY")
if not or_key:
    raise ValueError("OPENROUTER_API_KEY is not set")

client = OpenAI(
    api_key=or_key,
    base_url="https://openrouter.ai/api/v1",
)


# device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# load trained router checkpoint
checkpoint_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "router_sw.pt"
)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

dq = checkpoint["model_params"]["dq"]
dm = checkpoint["model_params"]["dm"]
num_models = checkpoint["model_params"]["num_models"]
router = MatrixFactorizationRouter(num_models, dq, dm).to(DEVICE)
router.load_state_dict(checkpoint["state_dict"])
router.eval()

# load model index
model_index_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train_model_index.json"
)
model_index = json.load(open(model_index_path))


def embed_query(query: str) -> np.ndarray:
    """Embed query using OpenAI API."""
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(emb.data[0].embedding, dtype=np.float32)

STRONG_MODEL = "gpt-4"
WEAK_MODEL = "gpt-3.5-turbo"

strong_id = int(model_index[STRONG_MODEL])
weak_id = int(model_index[WEAK_MODEL])

def get_strong_win_rate(prompt: str) -> float:
    q = embed_query(prompt)
    q = torch.tensor(q).unsqueeze(0).to(DEVICE)
    strong_tensor = torch.tensor([strong_id]).to(DEVICE)
    weak_tensor = torch.tensor([weak_id]).to(DEVICE)
    with torch.no_grad():
        p = router.win_probability(strong_tensor, weak_tensor, q)
    return float(p.item())



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)           # evals/question.jsonl
    p.add_argument("--out_csv", required=True)             # evals/my_router_winrates.csv
    args = p.parse_args()

    q = pd.read_json(args.questions, lines=True)
    rows = []
    for _, r in q.iterrows():
        turn1 = r["turns"][0]  # same as RouteLLM logic
        w = float(get_strong_win_rate(turn1))
        rows.append({"question_id": int(r["question_id"]), "strong_win_rate": w})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()