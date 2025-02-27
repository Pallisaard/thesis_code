import argparse
from pathlib import Path

import nibabel as nib
import torch
import tqdm
from monai.metrics.regression import MultiScaleSSIMMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average MS-SSIM between pairs of generated MRIs"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing generated MRI samples (sample_*.nii.gz files)",
    )
    parser.add_argument("--device", required=True, type=str, help="Device to use")
    parser.add_argument(
        "--output-file", required=True, type=str, help="Path to save the MS-SSIM scores"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[64, 256],
        required=True,
        help="Resolution of the MRI volumes (64 or 256)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("Computing MS-SSIM diversity scores")
    print(vars(args))

    # Setup metric with kernel size based on resolution
    kernel_size = 3 if args.resolution == 64 else 11
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, kernel_size=kernel_size)

    # Get all sample files
    sample_files = sorted(Path(args.input_dir).glob("sample_*.nii.gz"))
    print(f"Found {len(sample_files)} samples")

    # Get all unique pairs
    pairs = [
        (sample_files[i], sample_files[i + 1])
        for i in range(0, len(sample_files) - 1, 2)
    ]
    print(f"Computing MS-SSIM for {len(pairs)} pairs")

    ms_ssim_scores = []

    # Process pairs with progress bar
    pbar = tqdm.tqdm(pairs, desc="Computing pairwise MS-SSIM")
    for file1, file2 in pbar:
        # Load and normalize MRIs
        mri1 = nib.load(file1).get_fdata()  # type: ignore
        mri2 = nib.load(file2).get_fdata()  # type: ignore

        # Convert to tensors
        mri1 = torch.from_numpy(mri1).unsqueeze(0).unsqueeze(0).to(args.device)
        mri2 = torch.from_numpy(mri2).unsqueeze(0).unsqueeze(0).to(args.device)

        # Compute MS-SSIM
        score = ms_ssim(mri1, mri2)
        ms_ssim_scores.append(score.item())  # type: ignore

    # Calculate and print average
    average_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
    print(f"Average MS-SSIM: {average_ms_ssim:.4f}")

    # Save scores
    torch.save(torch.tensor(ms_ssim_scores), args.output_file)


if __name__ == "__main__":
    main()
