import argparse
from itertools import batched
from pathlib import Path

import nibabel as nib
import torch
import numpy as np
import tqdm

from thesis_code.dataloading.transforms import normalize_to
from thesis_code.metrics.utils import get_mri_vectorizer
from thesis_code.models.gans.hagan import LitHAGAN
from thesis_code.training.utils import numpy_to_nifti


def pars_args():
    parser = argparse.ArgumentParser(description="Generate n sampled MRIs")

    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--n-samples", required=True, type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "--checkpoint-path", required=True, type=str, help="Checkpoint path"
    )
    parser.add_argument("--device", type=str, help="Device to use", default="cpu")
    parser.add_argument(
        "--lambdas",
        type=float,
        help="Value for lambda_1 and lambda_2",
        default=1.0,
    )
    parser.add_argument("--batch-size", type=int, help="Batch size", default=4)

    return parser.parse_args()


def main():
    args = pars_args()

    # Load model
    print("Loading MRI vectorizer")
    mri_vectorizer = get_mri_vectorizer(50).eval().to(args.device)
    print("Loading model")
    model = (
        LitHAGAN.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path, latent_dim=1024
        )
        .eval()
        .to(args.device)
    )

    print("Generating vectorizer out array")
    mri_vectorizer_out = torch.zeros((args.n_samples, 512))

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    outer_bar = tqdm.tqdm(
        batched(range(args.n_samples), args.batch_size),
        total=args.n_samples // args.batch_size,
        desc="Generating samples",
    )

    for batch_ids in outer_bar:
        sample = model.safe_sample(len(batch_ids))
        sample = sample.detach().cpu().numpy()
        sample = 0.5 * sample + 0.5

        inner_bar = tqdm.tqdm(
            enumerate(batch_ids),
            total=len(batch_ids),
            desc="Saving samples",
            leave=False,
            position=1,
        )

        for i, sample_id in inner_bar:
            # Save MRI NIfTI sample
            sample_i = normalize_to(sample[i, 0], -1, 1)
            sample[i, 0] = sample_i
            sample_mri = numpy_to_nifti(sample_i)
            nib.save(sample_mri, f"{args.output_dir}/sample_{sample_id}.nii.gz")  # type: ignore

            # Get MRI vectorizer output
            mri_vectorizer_out[sample_id] = (
                mri_vectorizer(
                    torch.from_numpy(sample_i).unsqueeze(0).unsqueeze(0).to(args.device)
                )
                .detach()
                .cpu()
                .squeeze(0)
                .squeeze(0)
            )

    np.save(f"{args.output_dir}/mri_vectorizer_out.npy", mri_vectorizer_out)

    print("Finished generating samples")


if __name__ == "__main__":
    main()