import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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