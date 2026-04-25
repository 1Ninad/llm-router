import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
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
df = pd.read_parquet("data/dataset_chatbot_arena.parquet")
queries = df["query"].tolist()
embeddings = []

for q in tqdm(queries):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    )
    embeddings.append(emb.data[0].embedding)
embeddings = np.array(embeddings)

np.save("query_embeddings.npy", embeddings)
