#!/usr/bin/env python
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "Qwen/Qwen1.5-7B-Chat",                # Agent‑0
    "meta-llama/Meta-Llama-3-8B-Instruct", # Agent‑1
    "mistralai/Mistral-7B-Instruct",  # Agent‑2
]
DEVICE = 0          # CUDA gpu index; use ‑1 for CPU
MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
ANSWER_PROMPT = (
    "You are a medical expert. Below is a USMLE‑style MCQ.\n"
    "Respond with a short step‑by‑step explanation, then state only one option\n"
    "letter A‑E as your final answer in the exact format:\n"
    "Final answer: <letter>"
    "\n\nQuestion:\n{question}\n"
)

REVIEW_PROMPT = (
    "Here is a peer's answer (explanation + chosen option).\n"
    "----------------\n{peer_answer}\n----------------\n"
    "Your task:  ➊ Give a concise critique of their reasoning.\n"
    "            ➋ Rate how likely the answer is correct using **Score 1‑3**\n"
    "               (1 = incorrect, 2 = uncertain, 3 = very likely correct).\n"
    "            ➌ State your **Confidence 1‑5** in that score\n"
    "               (1 = low confidence, 5 = very sure).\n"
    "Respond with the template exactly:\n"
    "Review: <your comments>\nScore: <1‑3>\nConfidence: <1‑5>"
)

# ---------------------------------------------------------------------------
# MODEL HELPERS
# ---------------------------------------------------------------------------

def load_model(name: str) -> Pipeline:
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        device=DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )


def call_llm(pipe: Pipeline, prompt: str) -> str:
    out = pipe(prompt)[0]["generated_text"]
    return out[len(prompt) :].strip()

# ---------------------------------------------------------------------------
# PARSERS
# ---------------------------------------------------------------------------
LETTER_RE = re.compile(r"final answer\s*[:：]?\s*([A-E])", flags=re.I)
SCORE_RE = re.compile(r"score\s*[:：]?\s*(\d)", flags=re.I)
CONF_RE = re.compile(r"confidence\s*[:：]?\s*(\d)", flags=re.I)


def extract_letter(text: str) -> str:
    m = LETTER_RE.search(text)
    return m.group(1).upper() if m else "?"  # "?" means failed parse


def extract_score(text: str) -> int:
    m = SCORE_RE.search(text)
    return int(m.group(1)) if m else 2  # neutral


def extract_confidence(text: str) -> int:
    m = CONF_RE.search(text)
    return int(m.group(1)) if m else 3  # moderate

# ---------------------------------------------------------------------------
# VOTING (weighted by peer ratings × confidence)
# ---------------------------------------------------------------------------

def aggregate_vote(short_answers: List[str], peer_feedback: Dict[int, List[Tuple[int, int]]]) -> str:
    """peer_feedback[i] = list of (score, conf) given *to* i by peers."""
    weights = defaultdict(float)
    for agent_idx, ans in enumerate(short_answers):
        if ans == "?":
            continue
        # average of (score × confidence)
        if peer_feedback[agent_idx]:
            w = sum(s * c for s, c in peer_feedback[agent_idx]) / len(peer_feedback[agent_idx])
        else:
            w = 3.0  # no feedback; default moderate weight
        weights[ans] += w
    # highest weight wins; ties broken alphabetically
    return max(sorted(weights.items()), key=lambda kv: kv[1])[0]

# ---------------------------------------------------------------------------
# CORE PIPELINE FOR ONE QUESTION
# ---------------------------------------------------------------------------

def peer_review_round(question_str: str, pipes: List[Pipeline]):
    # 1️⃣  Independent answers
    long_answers: List[str] = []
    for pipe in pipes:
        long_answers.append(call_llm(pipe, ANSWER_PROMPT.format(question=question_str)))

    # 2️⃣  Cross‑reviews
    peer_fb: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # agent → list[(score,conf)]
    for r_idx, reviewer in enumerate(pipes):
        for a_idx, a_ans in enumerate(long_answers):
            if r_idx == a_idx:
                continue
            rev = call_llm(reviewer, REVIEW_PROMPT.format(peer_answer=a_ans))
            peer_fb[a_idx].append((extract_score(rev), extract_confidence(rev)))

    short_answers = [extract_letter(a) for a in long_answers]
    final_ans = aggregate_vote(short_answers, peer_fb)
    return final_ans, long_answers, short_answers, peer_fb

# ---------------------------------------------------------------------------
# DATASET UTILITIES
# ---------------------------------------------------------------------------

def load_medqa(split: str):
    """Return HF dataset split and detected field names."""
    ds = load_dataset("med_qa", "usmle", split=split)  # adjust if needed
    # Heuristics – MedQA typically has these keys
    q_key = next(k for k in ["question", "prompt", "text"] if k in ds.features)
    an_key = next(k for k in ["answer", "correct_answer", "label"] if k in ds.features)
    opt_key = next(
        (k for k in ["options", "choices", "options_list"] if k in ds.features),
        None,
    )
    return ds, q_key, an_key, opt_key


def stringify_question(item, q_key: str, opt_key: str | None) -> str:
    q = item[q_key].strip()
    if opt_key:
        opts = item[opt_key]
        # opts is list[str], map to A‑E
        letters = "ABCDE"[: len(opts)]
        opts_str = "\n".join(f"{l}. {t}" for l, t in zip(letters, opts))
        q = f"{q}\n\n{opts_str}"
    return q

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def run(max_samples: int, split: str, out_path: Path):
    ds, q_key, a_key, opt_key = load_medqa(split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print("Loading models… this may take a while")
    pipes = [load_model(n) for n in MODEL_NAMES]
    results = []
    correct = 0

    for idx, item in enumerate(ds):
        question_str = stringify_question(item, q_key, opt_key)
        gt = str(item[a_key]).strip().upper()
        final_ans, long_ans, short_ans, feedback = peer_review_round(question_str, pipes)
        if final_ans == gt:
            correct += 1

        results.append(
            {
                "idx": idx,
                "question": question_str,
                "ground_truth": gt,
                "agents": [
                    {
                        "model": MODEL_NAMES[i],
                        "long_answer": long_ans[i],
                        "short_answer": short_ans[i],
                    }
                    for i in range(3)
                ],
                "peer_feedback": {str(i): feedback[i] for i in feedback},
                "committee_answer": final_ans,
                "is_correct": final_ans == gt,
            }
        )
        print(
            f"[{idx}] voted {final_ans} (gt {gt}) — {'✓' if final_ans == gt else '✗'}"
        )

    acc = correct / len(results)
    print(f"\n=== COMMITTEE ACCURACY {correct}/{len(results)} = {acc:.2%} ===")

    # save JSONL
    with out_path.open("w", encoding="utf‑8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Results saved to {out_path.resolve()}")

# ---------------------------------------------------------------------------
# DIRECT EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # You can modify these parameters directly here
    max_samples = 20  # limit number of questions
    split = "validation"  # dataset split
    out_path = Path("peer_review_medqa.jsonl")  # output file
    
    run(max_samples, split, out_path)
