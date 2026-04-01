import torch
import json
import numpy as np
from openai import OpenAI
from matrix_factorization import MatrixFactorizationRouter
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# device
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print("Using device:", DEVICE)

# load router checkpoint
checkpoint = torch.load("router_sw.pt", map_location=DEVICE)

dq = checkpoint["model_params"]["dq"]
dm = checkpoint["model_params"]["dm"]
num_models = checkpoint["model_params"]["num_models"]

router = MatrixFactorizationRouter(num_models, dq, dm).to(DEVICE)
router.load_state_dict(checkpoint["state_dict"])
router.eval()
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
    prob = p.item()
    if prob > 0.75:
        chosen = strong_model
    else:
        chosen = weak_model

    return prob, chosen



# Normal prompts 
prompts = [
    "Explain what a hash table is",
    "What is machine learning in simple terms?",
    "Explain recursion with an example",
    "What is the capital of India?",
]

print("\n ROUTER TEST\n")
for prompt in prompts:
    prob, chosen = route(prompt, strong_model="gpt-4", weak_model="vicuna-13b")

    print("Prompt:", prompt)
    print("P(strong wins):", round(prob, 3))
    print("Router chooses:", chosen)


# Simulate routing to unseen LLMs – NOT WORKING:
print("\nRouting to unseen LLMs\n")

for prompt in prompts:
    prob, _ = route(
        prompt, strong_model="gpt-4", weak_model="vicuna-13b"
    )

    if prob > 0.75:
        chosen = "GPT-4.1 / Claude-3.5 / Gemini (strong model)"
    else:
        chosen = "Mixtral / Llama-3 8B (cheap model)"
    print("Prompt:", prompt)
    print("Router decision:", chosen)