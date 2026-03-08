# mapping_labels.py
import pandas as pd
import json

# ------------- Appendix A mapping -------------
appendix_tier_map = {
    "gpt-4": 0,
    "claude-instant-v1": 1,
    "claude-v1": 1,
    "gpt-3.5-turbo": 2,
    "vicuna-13b": 2,
    "guanaco-33b": 2,
    "koala-13b": 5,
    "gpt4all-13b-snoozy": 5,
    "palm-2": 4,
    "vicuna-7b": 4,
    "mpt-7b-chat": 6,
    "RWKV-4-Raven-14B": 6,
    "alpaca-13b": 6,
    "oasst-pythia-12b": 6,
    "fastchat-t5-3b": 7,
    "chatglm-6b": 7,
    "dolly-v2-12b": 8,
    "stablelm-tuned-alpha-7b": 8,
    "llama-13b": 9
}

DATA_PQ = "dataset_chatbot_arena.parquet"
OUT_PQ = "train_strong_weak.parquet"
MAPPING_OUT = "model_name_mapping_appendix.json"

def class_from_tier(t):
    if t is None:
        return None
    if t in (0,1):
        return "strong"
    if t == 2:
        return "weak"
    return "other"

def main():
    df = pd.read_parquet(DATA_PQ)
    # map tiers
    def get_tier(name):
        return appendix_tier_map.get(name, None)

    df["tier_a"] = df["model_a"].map(get_tier)
    df["tier_b"] = df["model_b"].map(get_tier)
    df["class_a"] = df["tier_a"].map(class_from_tier)
    df["class_b"] = df["tier_b"].map(class_from_tier)

    # keep only strong vs weak
    mask = ((df["class_a"]=="strong") & (df["class_b"]=="weak")) | ((df["class_b"]=="strong") & (df["class_a"]=="weak"))
    df_sw = df[mask].copy()

    def label_strong_won(row):
        if row["class_a"] == "strong" and row["class_b"] == "weak":
            if row["winner"] == "model_a":
                return 1
            if row["winner"] == "model_b":
                return 0
        if row["class_b"] == "strong" and row["class_a"] == "weak":
            if row["winner"] == "model_b":
                return 1
            if row["winner"] == "model_a":
                return 0
        return None

    df_sw["strong_won"] = df_sw.apply(label_strong_won, axis=1)
    df_sw = df_sw[df_sw["strong_won"].notnull()].reset_index(drop=True)

    # strong_model / weak_model columns
    df_sw["strong_model"] = df_sw.apply(lambda r: r["model_a"] if r["class_a"]=="strong" else r["model_b"], axis=1)
    df_sw["weak_model"] = df_sw.apply(lambda r: r["model_a"] if r["class_a"]=="weak" else r["model_b"], axis=1)

    train_df = df_sw[["query","strong_model","weak_model","strong_won"]].copy()
    train_df.to_parquet(OUT_PQ, index=False)

    # Save mapping for inspection
    mapping = {}
    unique_models = sorted(set(df["model_a"].unique()).union(set(df["model_b"].unique())))
    for m in unique_models:
        mapping[m] = {"tier": appendix_tier_map.get(m, None), "class": class_from_tier(appendix_tier_map.get(m, None))}
    with open(MAPPING_OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("WROTE", OUT_PQ, "rows =", len(train_df))
    print("WROTE", MAPPING_OUT, "for inspection. Unmapped models will have tier=None.")

if __name__ == "__main__":
    main()