import os
import zipfile
import shutil
import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def convert_dicom_to_nifti(dicom_folder, output_filepath):
    # Read DICOM files and sort them by instance number
    dicom_files = [pydicom.dcmread(str(fp)) for fp in Path(dicom_folder).glob("*.dcm")]
    dicom_files.sort(key=lambda x: int(x.InstanceNumber))

    # Stack DICOM images to create a 3D volume
    volume = np.stack([dcm.pixel_array for dcm in dicom_files])

    # Extract orientation information to build affine
    first_dcm = dicom_files[0]
    orientation = np.array(first_dcm.ImageOrientationPatient).reshape(2, 3)
    spacing = [
        float(first_dcm.PixelSpacing[0]),
        float(first_dcm.PixelSpacing[1]),
        float(first_dcm.SliceThickness),
    ]

    # Calculate the z-axis direction by taking the cross product of the x and y directions
    z_direction = np.cross(orientation[0], orientation[1])

    # Create the full 3x3 orientation matrix
    orientation_3x3 = np.vstack([orientation, z_direction])  # Shape is now (3, 3)

    # Calculate affine matrix based on DICOM orientation
    affine = np.eye(4)
    affine[:3, :3] = orientation_3x3 * spacing  # Applies scaling and orientation
    affine[:3, 3] = first_dcm.ImagePositionPatient  # Sets origin
    nifti_image = nib.Nifti1Image(volume, affine)  # type: ignore

    # Save NIfTI file
    nib.save(nifti_image, output_filepath)  # type: ignore


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

        for dicom_folder in tqdm(dicom_folders, desc="Converting DICOM to NIfTI"):
            output_filepath = output_dir / f"{dicom_folder.name}.nii.gz"
            # Convert DICOMs in this folder to a NIfTI file
            convert_dicom_to_nifti(dicom_folder, output_filepath)

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
