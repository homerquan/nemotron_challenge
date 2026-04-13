import argparse
import os
import torch
import re
from datasets import load_dataset

from unsloth import FastLanguageModel, is_bfloat16_supported

# Disable Unsloth's GRPO patch to avoid TorchDynamo tensor shape conflicts on Nemotron
import sys
if "trl.trainer.grpo_trainer" in sys.modules:
    del sys.modules["trl.trainer.grpo_trainer"]
if "trl.trainer" in sys.modules:
    del sys.modules["trl.trainer"]

from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from accelerate import Accelerator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich_argparse import RichHelpFormatter

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GRPO RL training using Unsloth-compatible model and Accelerate",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--base_model", type=str, default="/home/ubuntu/nemotron_model", help="Path to base model")
    parser.add_argument("--sft_adapter", type=str, default="/home/ubuntu/nemotron-reasoning-lora-new", help="Path to SFT adapter")
    parser.add_argument("--prm_adapter", type=str, default="/home/ubuntu/nemotron-generative-prm-lora-new", help="Path to PRM adapter")
    parser.add_argument("--output_dir", type=str, default="./nemotron-rl-grpo", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=5, help="Max steps (keep low for demo)")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--samples", type=int, default=1000, help="Number of MetaMathQA samples for RL")
    return parser.parse_args()

def extract_answer(text):
    matches = re.findall(r'\\boxed{((?:[^{}]+|{[^{}]*})*)}', text)
    if matches:
         return matches[-1]
    return None

def verify_correctness(answer, ground_truth):
    if not answer or not ground_truth:
        return 0.0
    ans_clean = answer.replace(" ", "").lower()
    gt_clean = ground_truth.replace(" ", "").lower()
    return 1.0 if ans_clean == gt_clean else 0.0

def main():
    args = parse_args()
    
    accelerator = Accelerator()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task1 = progress.add_task("[cyan]Loading SFT model and tokenizer...", total=None)
        
        # Load the base model pre-applied with the SFT adapter using Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.sft_adapter,
            max_seq_length=1024,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        progress.update(task1, completed=1, description="[green]SFT model loaded!")
        
        task2 = progress.add_task("[cyan]Loading PRM adapter...", total=None)
        try:
            model.load_adapter(args.prm_adapter, adapter_name="prm")
            model.set_adapter("default")
            progress.update(task2, completed=1, description="[green]PRM adapter loaded!")
        except Exception as e:
            progress.update(task2, description=f"[red]Error loading PRM adapter: {e}")
            console.print("[yellow]Make sure you have run the SFT and PRM training scripts first.")
            return

    def prm_reward_function(prompts, completions, **kwargs):
        rewards = []
        model.set_adapter("prm")
        model.eval()
        with torch.no_grad():
            for prompt, completion in zip(prompts, completions):
                raw_steps = [s.strip() for s in completion.split('\n') if s.strip()]
                if not raw_steps:
                    rewards.append(0.0)
                    continue
                step_scores = []
                prev_steps = ""
                for step in raw_steps:
                    prm_prompt = f"Evaluate the following reasoning step.\n\nQuestion: {prompt}\n\nPrevious Steps:\n{prev_steps}\n\nCandidate Step:\n{step}\n\nIs the Candidate Step correct? Output 1 for yes, 0 for no.\nLabel: "
                    inputs = tokenizer(prm_prompt, return_tensors="pt").to("cuda")
                    outputs = model(**inputs)
                    next_token_logits = outputs.logits[0, -1, :]
                    id_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
                    id_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
                    prob_1 = torch.softmax(torch.tensor([next_token_logits[id_0], next_token_logits[id_1]]), dim=0)[1].item()
                    step_scores.append(prob_1)
                    prev_steps += step + "\n"
                rewards.append(sum(step_scores) / len(step_scores))
        model.set_adapter("default")
        model.train()
        return rewards

    def outcome_reward_function(prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            ans = extract_answer(completion)
            rewards.append(0.5 if ans else 0.0)
        return rewards

    console.print(f"[cyan]Loading dataset for RL ({args.samples} samples)...")
    ds = load_dataset("meta-math/MetaMathQA", split=f"train[:{args.samples}]")
    def format_grpo(item):
        return {"prompt": f"Below is an instruction that describes a task. Write a response that appropriately completes the request. For math or reasoning tasks, always put the final answer in \\boxed{{}} format.\n\n### Instruction:\n{item['query']}\n\n### Response:\n"}
    ds = ds.map(format_grpo)

    training_args = GRPOConfig(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2, learning_rate=1e-5, logging_steps=1,
        max_steps=args.max_steps, save_steps=50, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        optim="paged_adamw_8bit", lr_scheduler_type="cosine", report_to="none",
        max_completion_length=256, num_generations=2
    )

    trainer = GRPOTrainer(
        model=model, args=training_args, train_dataset=ds,
        reward_funcs=[prm_reward_function, outcome_reward_function], processing_class=tokenizer,
    )

    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            console.print(f"[bold yellow]Resuming training from checkpoint: {last_checkpoint}[/bold yellow]")

    console.print("[bold yellow]Starting GRPO RL training...[/bold yellow]")
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
