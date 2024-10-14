import nibabel as nib
import sys


def analyze_single_nii_gz_file(filepath):
    """
    Analyzes a single NIfTI (.nii.gz) file to print its data type and resolution.

    Parameters
    ----------
    filepath : str
        Path to the NIfTI file.
    """
    try:
        img = nib.load(filepath)  # type: ignore
        data_type = img.get_data_dtype()  # type: ignore
        resolution = img.shape  # type: ignore

        print("File:", filepath)
        print("Data type:", data_type)
        print("Resolution:", resolution)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    analyze_single_nii_gz_file(file_path)
