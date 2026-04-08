import argparse
import os
import torch

from unsloth import FastLanguageModel, is_bfloat16_supported

from datasets import load_from_disk
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich_argparse import RichHelpFormatter

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ultimate SFT training using Unsloth and Accelerate",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--base_model", type=str, default="/home/ubuntu/nemotron_model", help="Path to the base model")
    parser.add_argument("--dataset_path", type=str, default="/home/ubuntu/reasoning_dataset_all", help="Path to the ultimate formatted dataset")
    parser.add_argument("--output_dir", type=str, default="./nemotron-ultimate-reasoning-lora", help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps (keep it low for demo)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    
    accelerator = Accelerator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task1 = progress.add_task("[cyan]Loading tokenizer and model...", total=None)
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=1024,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        progress.update(task1, completed=1, description="[green]Model and tokenizer loaded!")

        task2 = progress.add_task("[cyan]Loading ultimate dataset...", total=None)
        dataset = load_from_disk(args.dataset_path)
        progress.update(task2, completed=1, description=f"[green]Dataset loaded! Size: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        logging_steps=10,
        max_steps=args.max_steps,
        save_steps=100,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="paged_adamw_8bit",
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=1024
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            console.print(f"[bold yellow]Resuming training from checkpoint: {last_checkpoint}[/bold yellow]")

    console.print("[bold yellow]Starting ultimate training...[/bold yellow]")
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except (KeyboardInterrupt, Exception) as e:
        console.print(f"[bold red]Training interrupted! Saving training states as checkpoint to {args.output_dir}...[/bold red]")
        accelerator.save_state(os.path.join(args.output_dir, "accelerator_state_interrupt"))
        trainer.save_model(args.output_dir)
        trainer.save_state()
        raise e

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task3 = progress.add_task("[cyan]Saving final adapter...", total=None)
        trainer.save_model(args.output_dir)
        trainer.save_state()
        progress.update(task3, completed=1, description=f"[green]Done! Saved in {args.output_dir}")

if __name__ == "__main__":
    main()
