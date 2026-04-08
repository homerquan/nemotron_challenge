import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the evaluation results",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input csv (e.g. data/base_vllm_submission.csv)")
    parser.add_argument("--truth_csv", type=str, default="data/test.csv", help="Path to ground truth csv")
    parser.add_argument("--answer_col", type=str, default="base_answer", help="Column name containing generated answers")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        df_pred = pd.read_csv(args.input_csv)
        df_truth = pd.read_csv(args.truth_csv)
    except FileNotFoundError as e:
        console.print(f"[red]Error loading files: {e}")
        return

    # Check columns
    if args.answer_col not in df_pred.columns:
        console.print(f"[red]Column '{args.answer_col}' not found in {args.input_csv}")
        return
        
    correct = 0
    total = len(df_pred)
    
    # We assume 'id' links the two, or they are ordered identically
    for idx, row in df_pred.iterrows():
        pred_ans = str(row[args.answer_col]).strip()
        # Find truth
        if 'id' in row and 'id' in df_truth.columns:
            truth_row = df_truth[df_truth['id'] == row['id']]
            if len(truth_row) > 0:
                truth_ans = str(truth_row.iloc[0]['answer']).strip()
            else:
                continue
        else:
            # Fallback to index
            truth_ans = str(df_truth.iloc[idx]['answer']).strip()
            
        if pred_ans == truth_ans:
            correct += 1

    table = Table(title="Evaluation Results")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Samples", str(total))
    table.add_row("Correct", str(correct))
    table.add_row("Accuracy", f"{(correct/total)*100:.2f}%" if total > 0 else "0.00%")
    
    console.print(table)

if __name__ == "__main__":
    main()
