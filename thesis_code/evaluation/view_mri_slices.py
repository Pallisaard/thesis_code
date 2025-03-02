import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse


def view_mri_slices(nii_path):
    # Load the NIfTI file
    img = nib.load(nii_path)  # type: ignore
    data = img.get_fdata()  # type: ignore

    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Display the middle slice
    slice_idx = data.shape[2] // 2
    im = ax.imshow(data[slice_idx, :, :], cmap="gray")

    # Add a slider for slice navigation
    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(ax_slider, "Slice", 0, data.shape[2] - 1, valinit=slice_idx, valstep=1)

    def update(val):
        slice_idx = int(slider.val)
        im.set_data(data[slice_idx, :, :])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View MRI slices from a NIfTI file.")
    parser.add_argument("nii_path", type=str, help="Path to the NIfTI file")
    args = parser.parse_args()

    # Example usage
    view_mri_slices(args.nii_path)
