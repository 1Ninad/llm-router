import argparse
import numpy as np
import pandas as pd


def _trapz(y, x):
    # NumPy compatibility across versions
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def model_mean_score(questions: pd.DataFrame, judgements: pd.DataFrame, model_name: str) -> float:
    q = questions[["question_id"]].copy()
    q["routed_model"] = model_name
    r = q.merge(
        judgements,
        left_on=["question_id", "routed_model"],
        right_on=["question_id", "model"],
        how="left",
    )[["question_id", "model", "score"]]

    if r["score"].isna().any():
        missing_qids = r.loc[r["score"].isna(), "question_id"].drop_duplicates().tolist()[:10]
        raise ValueError(
            f"Missing judgements for model='{model_name}' on some question_ids. "
            f"Example missing IDs: {missing_qids}"
        )
    return float(r["score"].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True, help="Path to MT-Bench question.jsonl")
    p.add_argument("--judgements", required=True, help="Path to MT-Bench judgements.jsonl")
    p.add_argument("--winrates_csv", required=True, help="CSV with columns: question_id,strong_win_rate")
    p.add_argument("--strong_model", default="gpt-4")
    p.add_argument("--weak_model", default="gpt-3.5-turbo")
    p.add_argument("--num_results", type=int, default=10)
    p.add_argument("--method", default="my_router")
    args = p.parse_args()

    questions = pd.read_json(args.questions, lines=True)
    judgements = pd.read_json(args.judgements, lines=True)
    win = pd.read_csv(args.winrates_csv)

    needed_cols = {"question_id", "strong_win_rate"}
    if not needed_cols.issubset(set(win.columns)):
        raise ValueError(f"winrates_csv must contain columns {needed_cols}, got {set(win.columns)}")

    # enforce numeric
    win["strong_win_rate"] = pd.to_numeric(win["strong_win_rate"], errors="raise")
    if ((win["strong_win_rate"] < 0) | (win["strong_win_rate"] > 1)).any():
        raise ValueError("strong_win_rate must be in [0,1]")

    # one row per question_id
    if win["question_id"].duplicated().any():
        dups = win.loc[win["question_id"].duplicated(), "question_id"].unique().tolist()[:10]
        raise ValueError(f"Duplicate question_id(s) in winrates_csv, examples: {dups}")

    df = questions.merge(win, on="question_id", how="inner")
    if len(df) != len(questions):
        missing = sorted(set(questions["question_id"]) - set(df["question_id"]))
        raise ValueError(
            f"winrates rows mismatch: got {len(df)} vs questions {len(questions)}. "
            f"Missing question_ids example: {missing[:10]}"
        )

    strong_win_rates = df["strong_win_rate"].values
    _, thresholds = pd.qcut(strong_win_rates, args.num_results, retbins=True, duplicates="drop")

    rows = []
    for i, threshold in enumerate(thresholds):
        routed = np.where(
            (df["strong_win_rate"] >= threshold)
            if i != len(thresholds) - 1
            else (df["strong_win_rate"] > threshold),
            args.strong_model,
            args.weak_model,
        )

        q = df[["question_id"]].copy()
        q["routed_model"] = routed

        results = q.merge(
            judgements,
            left_on=["question_id", "routed_model"],
            right_on=["question_id", "model"],
            how="left",
        )[["question_id", "model", "score"]]

        if results["score"].isna().any():
            missing_qids = results.loc[results["score"].isna(), "question_id"].drop_duplicates().tolist()[:10]
            raise ValueError(
                "Missing judgement rows after merge. "
                f"Check model names. strong_model='{args.strong_model}', weak_model='{args.weak_model}'. "
                f"Example missing question_ids: {missing_qids}"
            )

        score = float(results["score"].mean())
        model_counts = results["model"].value_counts().to_dict()
        model_counts.setdefault(args.strong_model, 0)
        model_counts.setdefault(args.weak_model, 0)
        total = len(results)  # with no decontam: 80 * 2 = 160

        rows.append(
            {
                "method": args.method,
                "threshold": float(threshold),
                "strong_percentage": model_counts[args.strong_model] / total * 100.0,
                "accuracy": score,
                "total_rows": int(total),
            }
        )

    curve = pd.DataFrame(rows).sort_values("strong_percentage")
    auc = float(_trapz(curve["accuracy"], curve["strong_percentage"] / 100.0))

    weak_acc = model_mean_score(df, judgements, args.weak_model)
    strong_acc = model_mean_score(df, judgements, args.strong_model)

    weak_auc = float(_trapz(np.full(len(curve), weak_acc), curve["strong_percentage"] / 100.0))
    strong_auc = float(_trapz(np.full(len(curve), strong_acc), curve["strong_percentage"] / 100.0))

    denom = (strong_auc - weak_auc)
    if denom == 0:
        raise ValueError("Cannot compute APGR: strong_auc equals weak_auc (division by zero).")

    apgr = float((auc - weak_auc) / denom)

    print(f"method={args.method}")
    print(f"strong_model={args.strong_model}")
    print(f"weak_model={args.weak_model}")
    print(f"num_questions={len(df)}")
    print(f"rows_per_threshold={int(curve['total_rows'].iloc[0])}")
    print(f"weak_score={weak_acc:.6f}")
    print(f"strong_score={strong_acc:.6f}")
    print(f"AUC={auc:.6f}")
    print(f"APGR={apgr:.6f}")
    print(curve[["method", "threshold", "strong_percentage", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
