import argparse
import csv
import random
from datasets import load_dataset, Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich_argparse import RichHelpFormatter

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Format local dataset and MetaMathQA for training",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Path to input train.csv")
    parser.add_argument("--output_dir", type=str, default="data/reasoning_dataset", help="Output directory for the formatted dataset")
    parser.add_argument("--meta_math_samples", type=int, default=10000, help="Number of samples to take from MetaMathQA")
    return parser.parse_args()

def format_prompt(instruction, answer):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. For math or reasoning tasks, always put the final answer in \\boxed{{}} format.\n\n### Instruction:\n{instruction}\n\n### Response:\n{answer}"
    return prompt

def generate_dataset():
    args = parse_args()
    formatted_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task1 = progress.add_task("[cyan]Loading local train.csv...", total=1)
        csv_count = 0
        try:
            with open(args.train_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ans = str(row.get("answer", "")).strip()
                    prompt = row.get("prompt", "")
                    if not prompt or not ans:
                        continue
                    response = f"Based on the provided logic rules, the answer is \\boxed{{{ans}}}"
                    full_text = format_prompt(prompt, response)
                    formatted_data.append({"text": full_text})
                    csv_count += 1
            progress.update(task1, completed=1, description=f"[green]Loaded {csv_count} examples from {args.train_csv}")
        except Exception as e:
            progress.update(task1, description=f"[red]Error loading {args.train_csv}: {e}")

        task2 = progress.add_task("[cyan]Loading MetaMathQA dataset...", total=1)
        ds = load_dataset("meta-math/MetaMathQA", split=f"train[:{args.meta_math_samples}]")
        meta_count = 0
        
        for item in ds:
            response = item["response"]
            if "The answer is: " in response:
                parts = response.rsplit("The answer is: ", 1)
                answer_val = parts[1].strip().rstrip(".")
                new_response = parts[0] + f"The answer is: \\boxed{{{answer_val}}}"
            else:
                continue

            full_text = format_prompt(item["query"], new_response)
            formatted_data.append({"text": full_text})
            meta_count += 1

        progress.update(task2, completed=1, description=f"[green]Loaded {meta_count} examples from MetaMathQA")

        task3 = progress.add_task("[cyan]Saving dataset...", total=1)
        random.seed(42)
        random.shuffle(formatted_data)
        
        new_ds = Dataset.from_list(formatted_data)
        new_ds.save_to_disk(args.output_dir)
        progress.update(task3, completed=1, description=f"[green]Dataset saved to {args.output_dir} (Total: {len(formatted_data)})")

if __name__ == "__main__":
    generate_dataset()
