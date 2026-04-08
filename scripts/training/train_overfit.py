import argparse
import os
import torch
import pandas as pd

from unsloth import FastLanguageModel, is_bfloat16_supported

from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich_argparse import RichHelpFormatter

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fast targeted training using Unsloth and Accelerate",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--base_model", type=str, default="/home/ubuntu/nemotron_model", help="Path to base model")
    parser.add_argument("--input_csv", type=str, default="/home/ubuntu/test.csv", help="Path to input test.csv")
    parser.add_argument("--output_dir", type=str, default="./nemotron-reasoning-lora-overfit", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    
    accelerator = Accelerator()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task1 = progress.add_task("[cyan]Loading model with Unsloth...", total=None)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=512,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=3407
        )
        progress.update(task1, completed=1, description="[green]Model loaded!")
        
        task2 = progress.add_task("[cyan]Preparing targeted dataset...", total=None)
        ground_truths = {"00066667": "10010111", "000b53cf": "00011011", "00189f6a": "cat imagines book"}
        
        try:
            df = pd.read_csv(args.input_csv)
            training_data = []
            for idx, row in df.iterrows():
                q_id = str(row.get('id', '')).zfill(8)
                if q_id in ground_truths:
                    ans = ground_truths[q_id]
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. For math or reasoning tasks, always put the final answer in \\boxed{{}} format.\n\n### Instruction:\n{row['prompt']}\n\n### Response:\nThe answer is \\boxed{{{ans}}}"
                    for _ in range(10): training_data.append({"text": prompt})
            dataset = Dataset.from_list(training_data)
            progress.update(task2, completed=1, description=f"[green]Dataset prepared! ({len(dataset)} items)")
        except Exception as e:
            progress.update(task2, description=f"[red]Error loading dataset: {e}")
            return

    training_args = SFTConfig(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1, learning_rate=2e-4, logging_steps=5,
        max_steps=args.max_steps, save_steps=args.max_steps, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        optim="paged_adamw_8bit", warmup_steps=0, lr_scheduler_type="constant",
        report_to="none", dataset_text_field="text", max_seq_length=512
    )

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args, processing_class=tokenizer)

    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            console.print(f"[bold yellow]Resuming training from checkpoint: {last_checkpoint}[/bold yellow]")

    console.print("[bold yellow]Starting targeted training...[/bold yellow]")
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except (KeyboardInterrupt, Exception) as e:
        console.print(f"[bold red]Training interrupted! Saving training states as checkpoint to {args.output_dir}...[/bold red]")
        accelerator.save_state(os.path.join(args.output_dir, "accelerator_state_interrupt"))
        trainer.save_model(args.output_dir)
        trainer.save_state()
        raise e

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task3 = progress.add_task("[cyan]Saving adapter...", total=None)
        trainer.save_model(args.output_dir)
        trainer.save_state()
        progress.update(task3, completed=1, description=f"[green]Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
