import argparse
import numpy as np
import nibabel as nib
from itertools import batched
from pathlib import Path
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from scipy.ndimage import zoom


def reorient_nii_to_ras(nii):
    # Get the affine matrix
    affine = nii.affine

    # Desired RAS+ orientation
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))

    # Get the orientation of the image in matrix form
    current_ornt = nib.orientations.io_orientation(affine)

    # Get the transformation from current orientation to RAS+
    transform_ornt = nib.orientations.ornt_transform(current_ornt, ras_ornt)

    # Apply the orientation transform to the data
    reoriented_data = nib.orientations.apply_orientation(
        nii.get_fdata(), transform_ornt
    ).astype(np.float32)

    # Create a new NIfTI object with the reoriented data and updated affine
    new_affine = nii.affine @ nib.orientations.inv_ornt_aff(transform_ornt, nii.shape)
    reoriented_nii = nib.Nifti1Image(reoriented_data, new_affine)  # type: ignore

    return reoriented_nii


def resample_and_pad_nifti(
    nifti_img, target_voxel_size=(1.0, 1.0, 1.0), target_shape=(256, 256, 256)
):
    # Get the data and affine matrix
    data = nifti_img.get_fdata().astype(np.float32)
    affine = nifti_img.affine

    # Calculate the current voxel size from the affine matrix (diagonal elements)
    current_voxel_size = np.abs(np.diag(affine))[:3]

    # Calculate the resampling factors (current voxel size / target voxel size)
    resampling_factors = current_voxel_size / np.array(target_voxel_size)

    # Resample the data to the new voxel size using cubic interpolation (order=3)
    resampled_data = zoom(data, resampling_factors, order=3).astype(np.float32)

    # Determine the new affine matrix
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag(target_voxel_size)

    # Get the current shape of the resampled image
    resampled_shape = resampled_data.shape

    # Create an empty array for the padded/cropped output (target_shape)
    padded_data = np.zeros(target_shape, dtype=np.float32)

    # Compute cropping or padding indices
    min_shape = np.minimum(resampled_shape, target_shape)  # Minimum size for each axis
    crop_start = [(resampled_shape[i] - min_shape[i]) // 2 for i in range(3)]
    pad_start = [(target_shape[i] - min_shape[i]) // 2 for i in range(3)]

    # Crop or pad along each axis
    padded_data[
        pad_start[0] : pad_start[0] + min_shape[0],
        pad_start[1] : pad_start[1] + min_shape[1],
        pad_start[2] : pad_start[2] + min_shape[2],
    ] = resampled_data[
        crop_start[0] : crop_start[0] + min_shape[0],
        crop_start[1] : crop_start[1] + min_shape[1],
        crop_start[2] : crop_start[2] + min_shape[2],
    ]

    # Create a new NIfTI image with the padded/cropped data and the updated affine
    new_nifti = nib.Nifti1Image(padded_data, new_affine)  # type: ignore

    return new_nifti


def parse_args():
    parser = argparse.ArgumentParser(description="Reorient NIfTI files.")
    parser.add_argument("input_path", type=str, help="Path to the input NIfTI file.")
    parser.add_argument(
        "--single-file", action="store_true", help="Process a single file."
    )
    parser.add_argument(
        "--out-filename",
        type=str,
        default=None,
        help="Path to the output NIfTI file. If not provided, the input file will be overwritten.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to the number of CPU cores.",
    )
    parser.add_argument(
        "--reorient-only", action="store_true", help="Only reorient the NIfTI files."
    )
    parser.add_argument(
        "--reshape-only", action="store_true", help="Only reshape the NIfTI files."
    )
    args = parser.parse_args()
    assert not (args.reorient_only and args.reshape_only), "Cannot use both flags."
    return args


def process_batch(nii_files_batch, reorient=False, reshape=False):
    for nii_path in nii_files_batch:
        nii = nib.load(nii_path)  # type: ignore
        reoriented_nii = nii if reorient else reorient_nii_to_ras(nii)
        reshaped_nii = (
            reoriented_nii if reshape else resample_and_pad_nifti(reoriented_nii)
        )
        # save reorientation with same name as input
        output_path = nii_path.with_name(
            nii_path.stem.replace(".nii", "") + "_reoriented.nii.gz"
        )
        nib.save(reshaped_nii, output_path)  # type: ignore

    # Delete files in nii_files_batch
    for nii_path in nii_files_batch:
        nii_path.unlink()


def main():
    args = parse_args()

    if args.single_file:
        # For a single .nii.gz file
        nii_path = Path(args.input_path)
        nii = nib.load(nii_path)  # type: ignore
        reoriented_nii = nii if args.reshape_only else reorient_nii_to_ras(nii)
        reshaped_nii = (
            reoriented_nii
            if args.reorient_only
            else resample_and_pad_nifti(reoriented_nii)
        )
        # save reorientation with same name as input
        output_path = nii_path if args.out_filename is None else args.out_filename
        nib.save(reshaped_nii, output_path)  # type: ignore
        return 0

    # For all .nii.gz files in the input directory
    input_path = Path(args.input_path)
    nii_files = list(input_path.glob("*.nii.gz"))
    # Filter away files that already have been reoriented
    nii_files = [f for f in nii_files if not f.name.endswith("_reoriented.nii.gz")]
    batch_size = 5
    nii_files_batches = batched(nii_files, batch_size)
    n_batches = len(nii_files) // batch_size

    process_batch_partial = partial(
        process_batch, reshape=args.reorient_only, reorient=args.reshape_only
    )

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # executor.map returns an iterator
        # tqdm wraps the iterator to provide a progress bar
        # list forces the consumption of the iterator, ensuring tasks are
        # executed and progress bar updates
        list(
            tqdm(
                executor.map(process_batch_partial, nii_files_batches),
                total=n_batches,
                desc="Processing batches",
            )
        )

    # for nii_files_batch in tqdm(
    #     nii_files_batches, total=n_batches, desc="Processing batches"
    # ):
    #     for nii_path in tqdm(
    #         nii_files_batch, desc="Reorienting NIfTI files", leave=False
    #     ):
    #         nii = nib.load(nii_path)  # type: ignore
    #         reoriented_nii = reorient_nii_to_ras(nii)
    #         # save reorientation with same name as input
    #         output_path = nii_path.with_name(nii_path.stem + "_reoriented.nii.gz")
    #         nib.save(reoriented_nii, output_path)  # type: ignore

    #     for nii_path in tqdm(
    #         nii_files_batch, desc="Deleting original NIfTI files", leave=False
    #     ):
    #         nii_path.unlink()

    # Rename new files to original names
    output_paths = list(input_path.glob("*_reoriented.nii.gz"))
    for output_path in tqdm(output_paths, desc="Renaming reoriented files"):
        new_path = input_path / output_path.name.replace("_reoriented", "")
        output_path.rename(new_path)


if __name__ == "__main__":
    main()
