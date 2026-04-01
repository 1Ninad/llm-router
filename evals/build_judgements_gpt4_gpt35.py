import argparse
import json
import os
import re
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

STRONG_MODEL = "openai/gpt-4o"
WEAK_MODEL = "openai/gpt-4o-mini"   # cheaper weak
JUDGE_MODEL = "openai/gpt-4o-mini"  # cheaper judge


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


def parse_score(text: str) -> float:
    m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"\b([1-9](?:\.\d+)?|10(?:\.0+)?)\b", text)
    if m2:
        return float(m2.group(1))
    raise ValueError(f"Could not parse score from judge output: {text[:200]}")


def get_model_response(model_name: str, turns: list[str]) -> list[str]:
    # Turn 1
    r1 = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": turns[0]}],
        temperature=0,
    )
    a1 = r1.choices[0].message.content or ""

    # Turn 2 (multi-turn context)
    r2 = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": turns[0]},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": turns[1]},
        ],
        temperature=0,
    )
    a2 = r2.choices[0].message.content or ""
    return [a1, a2]


def judge_turn_score(question_id: int, model_name: str, turn: int, user_prompt: str, model_answer: str) -> float:
    prompt = f"""
You are an impartial judge. Rate the assistant answer quality from 1 to 10 considering correctness, helpfulness, relevance, and completeness.

Question (turn {turn}):
{user_prompt}

Assistant answer:
{model_answer}

Return:
1) one short explanation
2) final rating exactly in this format: [[n]]
"""
    jr = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    jtxt = jr.choices[0].message.content or ""
    return parse_score(jtxt)


def load_done_keys(out_path: str) -> set[tuple[int, str, int]]:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((int(row["question_id"]), row["model"], int(row["turn"])))
    return done


def append_row(out_fh, row: dict):
    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_fh.flush()
    os.fsync(out_fh.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="evals/question.jsonl")
    p.add_argument("--out", default="evals/judgements_gpt4_gpt35.jsonl")
    p.add_argument("--sleep_sec", type=float, default=0.0)
    args = p.parse_args()

    questions = pd.read_json(args.questions, lines=True)
    done = load_done_keys(args.out)
    total_needed = len(questions) * 2 * 2  # 320 for 80 questions

    print(f"loaded done rows: {len(done)}")
    print(f"target rows: {total_needed}")

    with open(args.out, "a", encoding="utf-8") as out_f:
        for _, r in questions.iterrows():
            qid = int(r["question_id"])
            turns = r["turns"]

            for model_name in (STRONG_MODEL, WEAK_MODEL):
                if (qid, model_name, 1) in done and (qid, model_name, 2) in done:
                    continue

                answers = get_model_response(model_name, turns)

                for turn_idx in (1, 2):
                    key = (qid, model_name, turn_idx)
                    if key in done:
                        continue

                    score = judge_turn_score(
                        question_id=qid,
                        model_name=model_name,
                        turn=turn_idx,
                        user_prompt=turns[turn_idx - 1],
                        model_answer=answers[turn_idx - 1],
                    )

                    row = {
                        "question_id": qid,
                        "model": model_name,
                        "score": float(score),
                        "turn": turn_idx,
                        "tstamp": time.time(),
                    }
                    append_row(out_f, row)
                    done.add(key)

                    if len(done) % 10 == 0:
                        print(f"progress: {len(done)}/{total_needed}")

                    if args.sleep_sec > 0:
                        time.sleep(args.sleep_sec)

    print(f"done. total_rows_in_file={len(done)} (expected {total_needed})")


if __name__ == "__main__":
    main()
