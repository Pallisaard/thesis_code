import argparse
from itertools import batched

import nibabel as nib
import torch
import numpy as np
import tqdm

from thesis_code.dataloading.transforms import normalize_to
from thesis_code.metrics.utils import get_mri_vectorizer
from thesis_code.models.gans.hagan import HAGAN
from thesis_code.training.utils import numpy_to_nifti


def pars_args():
    parser = argparse.ArgumentParser(description="Generate n sampled MRIs")

    parser.add_argument("output-dir", type=str, help="Output directory")
    parser.add_argument("n-samples", type=int, help="Number of samples to generate")
    parser.add_argument("checkpoint-path", type=int, help="Checkpoint path")
    parser.add_argument("device", type=str, help="Device to use", default="cpu")
    parser.add_argument(
        "lambdas", type=float, help="Value for lambda_1 and lambda_2", default=1.0
    )
    parser.add_argument("batch-size", type=int, help="Batch size", default=4)

    return parser.parse_args()


def main():
    args = pars_args()

    # Load model
    mri_vectorizer = get_mri_vectorizer(10).eval().to(args.device)
    model = (
        HAGAN.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
        .eval()
        .to(args.device)
    )

    mri_vectorizer_out = torch.zeros((args.n_samples, 512))

    outer_bar = tqdm.tqdm(
        batched(range(args.n_samples), args.batch_size),
        total=args.n_samples // args.batch_size,
        desc="Generating samples",
    )

    for batch_ids in outer_bar:
        print(f"Generating samples {batch_ids}")
        sample = model.safe_sample(len(batch_ids))
        sample = sample.detach().cpu().numpy()

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
            mri_vectorizer_out[i] = mri_vectorizer(
                torch.from_numpy(sample_i).to(args.device)
            )

    np.save(f"{args.output_dir}/mri_vectorizer_out.npy", mri_vectorizer_out)


if __name__ == "__main__":
    main()
