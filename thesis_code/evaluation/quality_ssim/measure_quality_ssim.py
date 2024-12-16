import argparse
from itertools import batched
from pathlib import Path

import nibabel as nib
import torch
import numpy as np
import tqdm
from monai.metrics.regression import SSIMMetric, MultiScaleSSIMMetric

from thesis_code.dataloading.transforms import normalize_to
from thesis_code.metrics.utils import get_mri_vectorizer
from thesis_code.models.gans.hagan import LitHAGAN
from thesis_code.training.utils import numpy_to_nifti


def pars_args():
    parser = argparse.ArgumentParser(
        description="Generate n sampled MRIs and compute SSIM between each and their most similar MRI in the vectorized file"
    )

    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--n-samples", required=True, type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "--checkpoint-path", required=True, type=str, help="Checkpoint path"
    )
    parser.add_argument("--data-dir", required=True, type=str, help="Data directory")
    parser.add_argument(
        "--vectorizer-file", required=True, type=str, help="Vectorized file"
    )
    parser.add_argument(
        "--filename-file", required=True, type=str, help="Filename file"
    )
    parser.add_argument(
        "--vector-dim", required=True, type=int, help="Vector dim", choices=[512, 2048]
    )
    parser.add_argument("--device", required=True, type=str, help="Device to use")
    parser.add_argument(
        "--lambdas", required=True, type=float, help="Value for lambda_1 and lambda_2"
    )
    parser.add_argument("--batch-size", type=int, help="Batch size", default=2)

    return parser.parse_args()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def find_most_similar_vector(
    vector: torch.Tensor, vectorizer_arr: torch.Tensor
) -> torch.Tensor:
    similarities = cosine_similarity(vector, vectorizer_arr)
    return torch.argmax(similarities)


def main():
    ## Generate n sampled MRIs, then find from the vectorized file and vectors the most similar MRI to each of the n sampled MRIs via cosine similarity, and compute SSIM. Average this across all n sampled MRIs.

    args = pars_args()
    model_id = 10 if args.vector_dim == 512 else 50

    # Make metric
    ssim = SSIMMetric(spatial_dims=3)

    # Load vectorizer
    mri_vectorizer = get_mri_vectorizer(model_id).eval().to(args.device)

    # Load model
    print("Loading model")
    model = (
        LitHAGAN.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            latent_dim=1024,
            lambda_1=args.lambdas,
            lambda_2=args.lambdas,
            use_dp_safe=True,
        )
        .eval()
        .to(args.device)
    )

    print("Getting vectorizer array")
    mri_vectorizer_arr = torch.load(args.vectorizer_file)

    print("Loading filenames")
    with open(args.filename_file, "r") as f:
        filenames = f.readlines()
    filenames = [f.strip() for f in filenames]

    if not Path(args.output_dir).exists():
        print("Creating output directory")
        Path(args.output_dir).mkdir(parents=True)

    outer_bar = tqdm.tqdm(
        batched(range(args.n_samples), args.batch_size),
        total=args.n_samples // args.batch_size,
        desc="Generating samples",
    )

    ssims = []

    for batch_ids in outer_bar:
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

            mri_vector = (
                mri_vectorizer(
                    torch.from_numpy(sample_i).unsqueeze(0).unsqueeze(0).to(args.device)
                )
                .detach()
                .cpu()
                .squeeze(0)
                .squeeze(0)
            )

            # Find most similar vector and the corresponding filename
            most_similar_idx = find_most_similar_vector(
                mri_vector,
                mri_vectorizer_arr,
            )
            most_similar_filename = filenames[most_similar_idx]

            # Compute SSIM
            most_similar_mri = nib.load(most_similar_filename).get_fdata()  # type: ignore
            most_similar_mri = normalize_to(most_similar_mri, -1, 1)
            y_pred = torch.from_numpy(sample_i).to(args.device)
            y = torch.from_numpy(most_similar_mri).to(args.device)
            ssim_val = ssim(y_pred, y)

            assert isinstance(ssim_val, torch.Tensor)
            ssims.append(ssim_val.detach().cpu())

    print("saving ssims")
    ssims = torch.stack(ssims)
    torch.save(ssims, f"{args.output_dir}/ssims.pt")

    print("Average SSIM:", ssims.mean().item())
