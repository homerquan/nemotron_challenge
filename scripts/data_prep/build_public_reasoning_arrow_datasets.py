
#!/usr/bin/env python3
"""
Build unified Arrow datasets for:
1) SFT/RL training      -> SFT_RL_DATASET/
2) synthetic PRM training -> PRM_DATASET/

Sources covered:
  1. Reasoning Gym (procedural generator library, not a fixed HF dataset)
  2. CohenQu/ARC-AGI-prompt
  3. sungyub/guru-logic-verl
  4. hzli1202/PuzzleWorld
  5. alexandrainst/multi-zebra-logic

Output:
  output_root/
    SFT_RL_DATASET/
    PRM_DATASET/

Notes:
- This script tries to normalize heterogeneous public datasets into a common format.
- The PRM dataset produced here is heuristic / synthetic unless the source already has
  step-level process annotations.
- Reasoning Gym is a procedural Python library. Install it with:
      pip install reasoning-gym
- Hugging Face datasets requires:
      pip install datasets pyarrow pandas

References used to choose sources:
- Reasoning Gym is a procedural library with 80+/100+ tasks and question/answer items.
- ARC-AGI-prompt is a public Hugging Face dataset with prompt-style rule-inference tasks.
- multi-zebra-logic is public on Hugging Face and Apache-2.0.
- guru-logic-verl is a public Apache-2.0 logic subset derived from GURU-RL-92k.
- PuzzleWorld is a 2025 public puzzle dataset on Hugging Face.
"""

import argparse
import json
import os
import random
import re
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


# -----------------------------
# Utility helpers
# -----------------------------
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def flatten_maybe_nested_answer(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        # Join simple lists nicely; fallback to JSON if nested.
        if all(isinstance(v, (str, int, float, bool)) for v in x):
            return "\n".join(str(v) for v in x)
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, dict):
        # common solution/answer containers
        for key in ["answer", "solution", "output", "target", "label", "final_answer"]:
            if key in x:
                return flatten_maybe_nested_answer(x[key])
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def maybe_split_dataset(ds_obj, preferred_splits: Tuple[str, ...] = ("train", "validation", "test")) -> Iterable[Tuple[str, Dataset]]:
    if isinstance(ds_obj, DatasetDict):
        for split in preferred_splits:
            if split in ds_obj:
                yield split, ds_obj[split]
        for split, ds in ds_obj.items():
            if split not in preferred_splits:
                yield split, ds
    else:
        yield "train", ds_obj


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text)).strip()


# -----------------------------
# Standardized row builders
# -----------------------------
def make_sft_rl_row(
    source: str,
    source_id: str,
    prompt: str,
    answer: str,
    split: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt = normalize_whitespace(prompt)
    answer = normalize_whitespace(answer)

    text = (
        "Below is an instruction that describes a reasoning task. "
        "Write a response that appropriately completes the request. "
        "When the task expects a single final answer, give the final answer clearly.\n\n"
        f"### Instruction:\n{prompt}\n\n"
        f"### Response:\n{answer}"
    )

    return {
        "id": source_id,
        "source": source,
        "split": split,
        "prompt": prompt,
        "answer": answer,
        "text": text,
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
    }


def prm_text_row(question: str, previous_steps: str, candidate_step: str, label: int) -> str:
    return (
        "Evaluate the following reasoning step.\n\n"
        f"Question: {question}\n\n"
        f"Previous Steps:\n{previous_steps}\n\n"
        f"Candidate Step:\n{candidate_step}\n\n"
        "Is the Candidate Step correct? Output 1 for yes, 0 for no.\n"
        f"Label: {int(label)}"
    )


# -----------------------------
# Synthetic PRM generation
# -----------------------------
def detect_task_type(prompt: str, answer: str, source: str) -> str:
    p = prompt.lower()
    a = answer.lower()

    if "grid" in p or "arc" in source.lower():
        return "grid_transform"
    if "zebra" in p or "clue" in p or "house" in p or "multi-zebra" in source.lower():
        return "logic_grid"
    if "binary" in p or "bit" in p or re.fullmatch(r"[01]{4,}", answer.strip()):
        return "binary"
    if "puzzle" in source.lower():
        return "puzzle"
    if re.fullmatch(r"-?\d+(\.\d+)?", answer.strip()):
        return "numeric"
    return "generic"


def perturb_binary_answer(ans: str) -> str:
    if not re.fullmatch(r"[01]+", ans):
        return "0"
    bits = list(ans)
    idx = random.randrange(len(bits))
    bits[idx] = "1" if bits[idx] == "0" else "0"
    return "".join(bits)


def perturb_numeric_answer(ans: str) -> str:
    try:
        x = float(ans)
        delta = max(abs(x) * 0.1, 1.0)
        return f"{x + random.choice([-1, 1]) * delta:.4f}".rstrip("0").rstrip(".")
    except Exception:
        return ans + "_wrong"


def perturb_text_answer(ans: str) -> str:
    words = ans.split()
    if not words:
        return "incorrect_answer"
    if len(words) == 1:
        w = words[0]
        if len(w) < 3:
            return w + "_x"
        i = random.randrange(len(w))
        c = "x" if not w[i].isalpha() else chr(((ord(w[i].lower()) - 97 + 1) % 26) + 97)
        return w[:i] + c + w[i + 1 :]
    words = words[:]
    random.shuffle(words)
    wrong = " ".join(words)
    return wrong if wrong != ans else wrong + " x"


def perturb_answer(ans: str, task_type: str) -> str:
    if task_type == "binary":
        return perturb_binary_answer(ans)
    if task_type == "numeric":
        return perturb_numeric_answer(ans)
    return perturb_text_answer(ans)


def build_prm_examples_from_standard_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    question = row["prompt"]
    answer = row["answer"]
    source = row["source"]
    task_type = detect_task_type(question, answer, source)
    wrong = perturb_answer(answer, task_type)

    positives = [
        ("", "First analyze the examples, clues, or constraints before committing to an answer.", 1),
        (
            "First analyze the examples, clues, or constraints before committing to an answer.",
            "Check that the proposed rule or interpretation is consistent with the information given in the prompt.",
            1,
        ),
        (
            "First analyze the examples, clues, or constraints before committing to an answer.\n"
            "Check that the proposed rule or interpretation is consistent with the information given in the prompt.",
            f"Therefore the final answer is {answer}.",
            1,
        ),
    ]

    negatives = [
        ("", "Ignore the examples or clues and guess the answer directly.", 0),
        (
            "First analyze the examples, clues, or constraints before committing to an answer.",
            "There is no need to verify whether the proposed rule matches the prompt.",
            0,
        ),
        (
            "First analyze the examples, clues, or constraints before committing to an answer.\n"
            "Check that the proposed rule or interpretation is consistent with the information given in the prompt.",
            f"Therefore the final answer is {wrong}.",
            0,
        ),
    ]

    if task_type == "grid_transform":
        positives.append((
            "First analyze the examples, clues, or constraints before committing to an answer.",
            "Infer the transformation pattern from the input-output demonstrations and apply it to the test input.",
            1,
        ))
        negatives.append((
            "First analyze the examples, clues, or constraints before committing to an answer.",
            "Copy one of the demonstration outputs directly without checking the new test input.",
            0,
        ))
    elif task_type == "logic_grid":
        positives.append((
            "First analyze the examples, clues, or constraints before committing to an answer.",
            "Use all clues together and avoid conclusions that contradict any clue.",
            1,
        ))
        negatives.append((
            "First analyze the examples, clues, or constraints before committing to an answer.",
            "Use only one clue and ignore the rest of the puzzle constraints.",
            0,
        ))
    elif task_type == "binary":
        negatives.append((
            "First analyze the examples, clues, or constraints before committing to an answer.",
            "Flip bits at random until the result looks plausible.",
            0,
        ))

    out = []
    for previous_steps, candidate_step, label in positives + negatives:
        out.append({
            "source_row_id": row["id"],
            "source": row["source"],
            "split": row["split"],
            "question": question,
            "previous_steps": previous_steps,
            "candidate_step": candidate_step,
            "label": int(label),
            "text": prm_text_row(question, previous_steps, candidate_step, int(label)),
            "metadata_json": row.get("metadata_json", "{}"),
        })
    return out


# -----------------------------
# Source loaders
# -----------------------------
def load_reasoning_gym_rows(
    size_per_task: int,
    seed: int,
    rg_tasks: List[str],
) -> List[Dict[str, Any]]:
    """
    Reasoning Gym is not a fixed HF dataset. It is a procedural Python library.
    We try to use reasoning_gym.create_dataset(task_name, size=..., seed=...).

    The public repo/docs show:
      dataset = reasoning_gym.create_dataset(dataset_name, **dataset_params)
      entries contain at least question, answer, metadata.
    """
    try:
        import reasoning_gym
    except ImportError as e:
        raise RuntimeError(
            "reasoning-gym is not installed. Please run: pip install reasoning-gym"
        ) from e

    rows: List[Dict[str, Any]] = []
    for task_name in rg_tasks:
        try:
            dataset = reasoning_gym.create_dataset(task_name, size=size_per_task, seed=seed)
            for idx, item in enumerate(dataset):
                q = safe_str(first_present(item, ["question", "prompt"]))
                a = flatten_maybe_nested_answer(first_present(item, ["answer", "solution", "target"]))
                if not q or not a:
                    continue
                source_id = f"reasoning_gym::{task_name}::{idx}"
                rows.append(
                    make_sft_rl_row(
                        source="reasoning_gym",
                        source_id=source_id,
                        prompt=q,
                        answer=a,
                        split="train",
                        metadata={
                            "task_name": task_name,
                            "raw_item": item,
                        },
                    )
                )
        except Exception as e:
            print(f"[WARN] reasoning-gym task failed: {task_name} -> {e}")
    return rows


def normalize_arc_prompt_row(example: Dict[str, Any], source: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    prompt = safe_str(first_present(example, ["prompt", "question", "instruction", "text"]))
    answer = flatten_maybe_nested_answer(first_present(example, ["answer", "solution", "output", "target", "completion"]))
    if not prompt:
        return None
    if not answer:
        # Many prompt-style datasets still need a target field. If not found, skip.
        return None
    source_id = safe_str(first_present(example, ["id", "task_id", "uuid"], default=f"{source}::{split}::{idx}"))
    return make_sft_rl_row(source, source_id, prompt, answer, split, metadata={"raw_item": example})


def normalize_zebra_row(example: Dict[str, Any], source: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    prompt = safe_str(first_present(example, ["question", "prompt", "instruction", "text"]))
    answer = flatten_maybe_nested_answer(first_present(example, ["solution", "answer", "output", "target", "label"]))
    if not prompt or not answer:
        # Fallback: build from clues/context.
        clues = first_present(example, ["clues", "context", "puzzle"])
        if clues is not None and answer:
            prompt = "Solve the following zebra-style logic puzzle.\n\n" + safe_str(clues)
    if not prompt or not answer:
        return None
    source_id = safe_str(first_present(example, ["id", "task_id", "uuid"], default=f"{source}::{split}::{idx}"))
    return make_sft_rl_row(source, source_id, prompt, answer, split, metadata={"raw_item": example})


def normalize_puzzleworld_row(example: Dict[str, Any], source: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    prompt = safe_str(first_present(example, ["question", "prompt", "problem", "instruction", "text"]))
    answer = flatten_maybe_nested_answer(first_present(example, ["answer", "solution", "final_answer", "output", "target"]))
    if not prompt or not answer:
        return None
    source_id = safe_str(first_present(example, ["id", "task_id", "uuid"], default=f"{source}::{split}::{idx}"))
    return make_sft_rl_row(source, source_id, prompt, answer, split, metadata={"raw_item": example})


def normalize_guru_logic_row(example: Dict[str, Any], source: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    prompt = safe_str(first_present(example, ["prompt", "question", "instruction", "query", "problem", "text"]))
    answer = flatten_maybe_nested_answer(first_present(example, ["answer", "ground_truth", "solution", "output", "target", "label"]))
    if not prompt:
        conversations = first_present(example, ["messages", "conversation"])
        if isinstance(conversations, list) and conversations:
            prompt = safe_str(conversations[0])
    if not prompt or not answer:
        return None
    source_id = safe_str(first_present(example, ["id", "task_id", "uuid"], default=f"{source}::{split}::{idx}"))
    return make_sft_rl_row(source, source_id, prompt, answer, split, metadata={"raw_item": example})


def load_hf_rows(
    dataset_id: str,
    normalizer,
    max_rows: int = 0,
    source_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    source_name = source_name or dataset_id
    ds_obj = load_dataset(dataset_id)
    rows: List[Dict[str, Any]] = []
    for split, ds in maybe_split_dataset(ds_obj):
        for idx, ex in enumerate(ds):
            if max_rows and len(rows) >= max_rows:
                return rows
            try:
                row = normalizer(ex, source_name, split, idx)
                if row:
                    rows.append(row)
            except Exception as e:
                print(f"[WARN] normalize failure for {source_name}/{split}/{idx}: {e}")
    return rows


# -----------------------------
# Main builder
# -----------------------------
def build_all_rows(
    output_root: str,
    seed: int,
    rg_size_per_task: int,
    rg_tasks: List[str],
    hf_max_rows_per_source: int,
    include_sources: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)

    sft_rl_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    if "reasoning_gym" in include_sources:
        try:
            sft_rl_rows.extend(
                load_reasoning_gym_rows(
                    size_per_task=rg_size_per_task,
                    seed=seed,
                    rg_tasks=rg_tasks,
                )
            )
            print(f"[OK] reasoning_gym -> {sum(1 for r in sft_rl_rows if r['source']=='reasoning_gym')} rows")
        except Exception as e:
            errors.append(f"reasoning_gym: {e}\n{traceback.format_exc()}")

    hf_sources = [
        ("CohenQu/ARC-AGI-prompt", normalize_arc_prompt_row, "arc_agi_prompt"),
        ("sungyub/guru-logic-verl", normalize_guru_logic_row, "guru_logic_verl"),
        ("hzli1202/PuzzleWorld", normalize_puzzleworld_row, "puzzleworld"),
        ("alexandrainst/multi-zebra-logic", normalize_zebra_row, "multi_zebra_logic"),
    ]

    for dataset_id, normalizer, source_name in hf_sources:
        if source_name not in include_sources and dataset_id not in include_sources:
            continue
        try:
            rows = load_hf_rows(
                dataset_id=dataset_id,
                normalizer=normalizer,
                max_rows=hf_max_rows_per_source,
                source_name=source_name,
            )
            sft_rl_rows.extend(rows)
            print(f"[OK] {dataset_id} -> {len(rows)} rows")
        except Exception as e:
            errors.append(f"{dataset_id}: {e}\n{traceback.format_exc()}")

    if errors:
        err_path = os.path.join(output_root, "build_errors.log")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(errors))
        print(f"[WARN] Some sources failed. See: {err_path}")

    prm_rows: List[Dict[str, Any]] = []
    for row in sft_rl_rows:
        prm_rows.extend(build_prm_examples_from_standard_row(row))

    return sft_rl_rows, prm_rows


def save_arrow_dataset(rows: List[Dict[str, Any]], out_dir: str) -> None:
    ensure_dir(out_dir)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(out_dir)
    print(f"[SAVED] {out_dir} ({len(ds)} rows)")


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[SAVED] {path} ({len(rows)} rows)")


def parse_args():
    parser = argparse.ArgumentParser(description="Download/normalize public reasoning datasets into Arrow SFT_RL_DATASET and PRM_DATASET")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Reasoning Gym controls
    parser.add_argument("--rg_size_per_task", type=int, default=200)
    parser.add_argument(
        "--rg_tasks",
        nargs="+",
        default=["countdown", "cryptarithm", "letter_jumble", "chain_sum", "graph_color"],
        help="Reasoning Gym task names to sample if reasoning-gym is installed."
    )

    # Hugging Face controls
    parser.add_argument("--hf_max_rows_per_source", type=int, default=3000)

    # Which sources to include
    parser.add_argument(
        "--include_sources",
        nargs="+",
        default=[
            "reasoning_gym",
            "arc_agi_prompt",
            "guru_logic_verl",
            "puzzleworld",
            "multi_zebra_logic",
        ],
        help="Subset of sources to include.",
    )

    # Optional jsonl exports
    parser.add_argument("--save_jsonl", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_root)

    sft_rl_rows, prm_rows = build_all_rows(
        output_root=args.output_root,
        seed=args.seed,
        rg_size_per_task=args.rg_size_per_task,
        rg_tasks=args.rg_tasks,
        hf_max_rows_per_source=args.hf_max_rows_per_source,
        include_sources=args.include_sources,
    )

    sft_out = os.path.join(args.output_root, "SFT_RL_DATASET")
    prm_out = os.path.join(args.output_root, "PRM_DATASET")

    save_arrow_dataset(sft_rl_rows, sft_out)
    save_arrow_dataset(prm_rows, prm_out)

    if args.save_jsonl:
        save_jsonl(sft_rl_rows, os.path.join(args.output_root, "sft_rl_dataset.jsonl"))
        save_jsonl(prm_rows, os.path.join(args.output_root, "prm_dataset.jsonl"))

    print("\nDone.")
    print(f"SFT/RL rows: {len(sft_rl_rows)}")
    print(f"PRM rows:    {len(prm_rows)}")
    print("\nExample usage:")
    print(f"python {os.path.basename(__file__)} --output_root ./public_reasoning_data --save_jsonl")


if __name__ == "__main__":
    main()
