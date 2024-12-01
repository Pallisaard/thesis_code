import os
import zipfile
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import dicom2nifti


def process_zip_files(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert glob result to a list to get the total number of zip files
    zip_files = list(input_dir.glob("*.zip"))

    # Iterate through all zip files in the input directory
    for zip_path in tqdm(zip_files, desc="Processing zip files"):
        # Unpack the zip file in a temporary directory
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            unpacked_folder = input_dir / zip_path.stem
            zip_ref.extractall(unpacked_folder)

        dicom_folders = []

        # Walk through the unpacked folder
        for root, _, files in os.walk(unpacked_folder):
            if any(file.endswith(".dcm") for file in files):
                # We have reached a directory with DICOM files
                dicom_folders.append(Path(root))

        for dicom_folder in tqdm(
            dicom_folders, desc="Converting DICOM to NIfTI", leave=False
        ):
            output_filepath = output_dir / f"{dicom_folder.name}.nii.gz"
            # Convert DICOMs in this folder to a NIfTI file
            dicom2nifti.dicom_series_to_nifti(
                dicom_folder, output_filepath, reorient_nifti=True
            )
            # convert_dicom_to_nifti(dicom_folder, output_filepath)

        # Delete the unpacked folder after processing
        shutil.rmtree(unpacked_folder)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DICOM zip archives to NIfTI files."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input directory containing zip files with DICOM data.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory where NIfTI files will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_zip_files(args.input_dir, args.output_dir)
