import argparse
from pathlib import Path
import shutil

from tqdm import tqdm


def copy_unique_niftis(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    copied_users = set()
    nifti_files = list(input_path.glob("*.nii.gz"))

    for nifti_file in tqdm(nifti_files, desc="Copying files"):
        user = nifti_file.stem.split("-")[0]
        if user not in copied_users:
            shutil.copy(nifti_file, output_path / nifti_file.name)
            copied_users.add(user)


def main():
    parser = argparse.ArgumentParser(
        description="Copy unique nifti files based on user."
    )
    parser.add_argument(
        "--input-dir", type=str, help="Directory containing the nifti files."
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to copy unique nifti files to."
    )

    args = parser.parse_args()

    copy_unique_niftis(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
