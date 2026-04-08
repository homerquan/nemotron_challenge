import argparse
import random
import re
from datasets import load_dataset, Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich_argparse import RichHelpFormatter

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Process Reward Model dataset",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--output_dir", type=str, default="./generative_prm_dataset", help="Output directory for the PRM dataset")
    parser.add_argument("--samples", type=int, default=5000, help="Number of MetaMathQA samples to use")
    return parser.parse_args()

def corrupt_step(step_text):
    mutated = step_text
    if '+' in mutated and random.random() < 0.5:
        mutated = mutated.replace('+', '-', 1)
    elif '-' in mutated and random.random() < 0.5:
        mutated = mutated.replace('-', '+', 1)
    elif '*' in mutated and random.random() < 0.5:
        mutated = mutated.replace('*', '/', 1)
    else:
        numbers = re.findall(r'\d+', mutated)
        if numbers:
            num = random.choice(numbers)
            action = random.choice([1, -1, 10])
            if action == 10:
                new_num = str(int(num) * 10)
            else:
                new_num = str(int(num) + action)
            mutated = mutated.replace(num, new_num, 1)
        else:
            mutated = mutated + " Therefore, the logic is broken."
    return mutated

def generate_prm_dataset():
    args = parse_args()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
    
        task1 = progress.add_task("[cyan]Loading MetaMathQA...", total=1)
        ds = load_dataset("meta-math/MetaMathQA", split=f"train[:{args.samples}]")
        progress.update(task1, completed=1, description="[green]Loaded MetaMathQA")
        
        task2 = progress.add_task("[cyan]Extracting steps...", total=len(ds))
        prm_data = []
        
        for item in ds:
            query = item['query']
            response = item['response']
            
            raw_steps = [s.strip() for s in response.split('\n') if s.strip()]
            if len(raw_steps) == 1:
                raw_steps = [s.strip() + '.' for s in response.split('.') if s.strip()]
                
            if not raw_steps:
                progress.advance(task2)
                continue
                
            prev_steps = ""
            for step in raw_steps:
                prompt = f"Evaluate the following reasoning step.\n\nQuestion: {query}\n\nPrevious Steps:\n{prev_steps}\n\nCandidate Step:\n{step}\n\nIs the Candidate Step correct? Output 1 for yes, 0 for no.\nLabel: "
                prm_data.append({"text": prompt + "1"})
                
                bad_step = corrupt_step(step)
                if bad_step != step:
                    bad_prompt = f"Evaluate the following reasoning step.\n\nQuestion: {query}\n\nPrevious Steps:\n{prev_steps}\n\nCandidate Step:\n{bad_step}\n\nIs the Candidate Step correct? Output 1 for yes, 0 for no.\nLabel: "
                    prm_data.append({"text": bad_prompt + "0"})
                
                prev_steps += step + "\n"
            progress.advance(task2)
            
        task3 = progress.add_task(f"[cyan]Saving {len(prm_data)} examples...", total=1)
        random.shuffle(prm_data)
        new_ds = Dataset.from_list(prm_data)
        new_ds.save_to_disk(args.output_dir)
        progress.update(task3, completed=1, description=f"[green]Saved PRM dataset to {args.output_dir}")

if __name__ == "__main__":
    random.seed(42)
    generate_prm_dataset()
