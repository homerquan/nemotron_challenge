import argparse
import pandas as pd
import re
import gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from rich.console import Console
from rich.progress import track
from rich_argparse import RichHelpFormatter

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA model using vLLM",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--base_model", type=str, default="/home/ubuntu/nemotron_model", help="Path to base model")
    parser.add_argument("--lora_path", type=str, default="./nemotron-reasoning-lora-final", help="Path to LoRA adapter")
    parser.add_argument("--input_csv", type=str, default="data/test.csv", help="Path to input test data")
    parser.add_argument("--output_csv", type=str, default="data/lora_vllm_submission.csv", help="Output predictions")
    parser.add_argument("--max_tokens", type=int, default=7680, help="Max new tokens to generate")
    return parser.parse_args()

def extract_answer(response_text):
    matches = re.findall(r'\\boxed{((?:[^{}]+|{[^{}]*})*)}', response_text)
    if matches:
         return matches[-1]
    return "NOT_FOUND"

def main():
    args = parse_args()
    
    console.print(f"[cyan]Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    console.print(f"[cyan]Loading base model and LoRA into vLLM...")
    try:
        llm = LLM(
            model=args.base_model,
            enable_lora=True,
            max_lora_rank=32,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            dtype="bfloat16",
        )
    except Exception as e:
        console.print(f"[red]CRITICAL ERROR loading model: {e}")
        return

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    
    prompts = []
    for idx, row in df.iterrows():
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. For math or reasoning tasks, always put the final answer in \\boxed{{}} format.\n\n### Instruction:\n{row['prompt']}\n\n### Response:\n"
        prompts.append(prompt)

    console.print(f"[yellow]Generating answers for {len(prompts)} prompts...")
    outputs = llm.generate(
        prompts, 
        sampling_params,
        lora_request=LoRARequest("my_lora", 1, args.lora_path)
    )
    
    answers = []
    raw_texts = []
    for output in track(outputs, description="[cyan]Extracting answers..."):
        generated_text = output.outputs[0].text
        ans = extract_answer(generated_text)
        answers.append(ans)
        raw_texts.append(generated_text)

    df['lora_answer'] = answers
    df['raw_text'] = raw_texts
    df.to_csv(args.output_csv, index=False)
    console.print(f"[bold green]Done! Evaluation answers saved to {args.output_csv}")
    
    del llm
    gc.collect()

if __name__ == "__main__":
    main()
