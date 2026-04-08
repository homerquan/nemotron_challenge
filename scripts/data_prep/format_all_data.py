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
        description="Format massive combined dataset for training",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Path to input train.csv")
    parser.add_argument("--output_dir", type=str, default="data/reasoning_dataset_all", help="Output directory")
    parser.add_argument("--meta_math_samples", type=int, default=10000, help="Number of MetaMathQA samples")
    parser.add_argument("--numina_samples", type=int, default=10000, help="Number of NuminaMath samples")
    return parser.parse_args()

def format_prompt(instruction, answer):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. For math or reasoning tasks, always put the final answer in \\boxed{{}} format.\n\n### Instruction:\n{instruction}\n\n### Response:\n{answer}"
    return prompt

def generate_dataset():
    args = parse_args()
    all_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task1 = progress.add_task("[cyan]Loading local train.csv...", total=1)
        try:
            with open(args.train_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ans = str(row.get('answer', '')).strip()
                    prompt = row.get('prompt', '')
                    if not prompt or not ans: continue
                    response = f"Based on the provided logic rules, the answer is \\boxed{{{ans}}}"
                    all_data.append({"text": format_prompt(prompt, response)})
            progress.update(task1, completed=1, description=f"[green]Added {args.train_csv}")
        except Exception as e: 
            progress.update(task1, description=f"[red]Error: {e}")

        task2 = progress.add_task("[cyan]Loading MetaMathQA...", total=1)
        try:
            ds = load_dataset("meta-math/MetaMathQA", split=f"train[:{args.meta_math_samples}]")
            for item in ds:
                response = item['response']
                if "The answer is: " in response:
                    parts = response.rsplit("The answer is: ", 1)
                    answer_val = parts[1].strip().rstrip('.')
                    new_response = parts[0] + f"The answer is: \\boxed{{{answer_val}}}"
                    all_data.append({"text": format_prompt(item['query'], new_response)})
            progress.update(task2, completed=1, description="[green]Added MetaMathQA")
        except Exception as e: 
            progress.update(task2, description=f"[red]Error: {e}")

        task3 = progress.add_task("[cyan]Loading NuminaMath...", total=1)
        try:
            ds = load_dataset("PrimeIntellect/NuminaMath-QwQ-CoT-5M", split=f"train[:{args.numina_samples}]")
            count = 0
            for item in ds:
                msgs = item.get('messages', [])
                if msgs and len(msgs) >= 2:
                    prompt = msgs[0].get('content', '')
                    response = msgs[1].get('content', '')
                    if "\\boxed" in response:
                        all_data.append({"text": format_prompt(prompt, response)})
                        count += 1
                else:
                    response = item.get('response', '')
                    prompt = item.get('prompt', '')
                    if response and prompt and "\\boxed" in response:
                        all_data.append({"text": format_prompt(prompt, response)})
                        count += 1
            progress.update(task3, completed=1, description=f"[green]Added {count} NuminaMath examples")
        except Exception as e: 
            progress.update(task3, description=f"[red]Error: {e}")

        task4 = progress.add_task("[cyan]Saving dataset...", total=1)
        random.seed(42)
        random.shuffle(all_data)
        
        new_ds = Dataset.from_list(all_data)
        new_ds.save_to_disk(args.output_dir)
        progress.update(task4, completed=1, description=f"[green]Saved {len(all_data)} items to {args.output_dir}")

if __name__ == "__main__":
    generate_dataset()
