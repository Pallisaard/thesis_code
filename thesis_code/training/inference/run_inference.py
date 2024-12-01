import argparse
import nibabel as nib
import numpy as np
from thesis_code.training.utils import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Model arguments.
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["cicek_3d_vae_64", "cicek_3d_vae_256", "kwon_gan"],
        help="Name of the model to train.",
    )

    parser.add_argument(
        "--checkpoint-path", required=True, type=str, help="Path to model checkpoint."
    )

    parser.add_argument(
        "--output-path",
        required=True,
        type=str,
        help="Output path for the generated samples.",
    )

    parser.add_argument("--num-inference")

    return parser.parse_args()


def save_as_nii(array: np.ndarray, output_path: str):
    img = nib.Nifti1Image(array, affine=np.eye(4))  # type: ignore
    nib.save(img, output_path)  # type: ignore


def main():
    args = parse_args()

    model = get_model(
        model_name=args.model_name,
        latent_dim=None,
        checkpoint_path=args.checkpoint_path,
    )
    model.eval()

    samples = model.sample(args.num_inference).detach().cpu().numpy()

    for i, sample in enumerate(samples):
        output_file = f"{args.output_path}/sample_{i}.nii.gz"
        save_as_nii(sample, output_file)


if __name__ == "__main__":
    main()
