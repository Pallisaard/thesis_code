import argparse
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect MS-SSIM scores from multiple directories into a DataFrame"
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        type=str,
        help="Base directory containing model output folders",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=str,
        help="Path to save the CSV file with collected scores",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    
    # Find all msssim_scores.pt files
    score_files = list(base_dir.glob("**/msssim_scores.pt"))
    print(f"Found {len(score_files)} score files")
    
    # Collect data for DataFrame
    data = []
    for score_file in score_files:
        # Get model name from parent directory
        model_name = score_file.parent.name.replace("generated-examples-", "")
        
        # Load scores
        scores = torch.load(score_file)
        avg_score = scores.mean().item()
        
        data.append({
            "model": model_name,
            "ms_ssim": avg_score
        })
    
    # Create DataFrame and sort by model name
    df = pd.DataFrame(data)
    df = df.sort_values("model")
    
    # Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"Saved scores to {args.output_file}")
    
    # Print summary
    print("\nMS-SSIM Scores Summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main() 