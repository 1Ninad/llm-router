import pandas as pd
import numpy as np
import json

df = pd.read_parquet("dataset_chatbot_arena.parquet")
models = list(set(df["model_a"]).union(set(df["model_b"])))
model_to_id = {m:i for i,m in enumerate(models)}

df["model_a_id"] = df["model_a"].map(model_to_id)
df["model_b_id"] = df["model_b"].map(model_to_id)

winner_a = (df["winner"] == "model_a").astype(int)

np.save("winner_a.npy", winner_a.values)
np.save("model_a_ids.npy", df["model_a_id"].values)
np.save("model_b_ids.npy", df["model_b_id"].values)

json.dump(model_to_id, open("model_index.json","w"))