import napari
import nibabel as nib
import argparse


def view_mri_in_3d(nii_paths):
    # Create a Napari viewer
    viewer = napari.Viewer()

    for nii_path in nii_paths:
        # Load the NIfTI file
        img = nib.load(nii_path)  # type: ignore
        data = img.get_fdata()  # type: ignore

        # Add the MRI volume to the viewer
        viewer.add_image(
            data, name=f"MRI: {nii_path}", rendering="attenuated_mip"
        )  # Use "attenuated_mip" for 3D rendering

    # Start the Napari event loop
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View MRI volumes in 3D using Napari.")
    parser.add_argument(
        "nii_paths", type=str, nargs="+", help="Paths to the NIfTI files"
    )
    args = parser.parse_args()

    # Example usage
    view_mri_in_3d(args.nii_paths)
