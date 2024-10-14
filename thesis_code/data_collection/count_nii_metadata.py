import nibabel as nib
import os
from collections import Counter
from tqdm import tqdm


def analyze_nii_gz_files(directory):
    """
    Analyzes NIfTI (.nii.gz) files in a directory to count different data
    types and resolutions.

    Parameters
    ----------
    directory : str
        Path to the directory containing the NIfTI files.
    """

    data_types = Counter()
    resolutions = Counter()
    files = [f for f in os.listdir(directory) if f.endswith(".nii.gz")]

    for filename in tqdm(files, desc="Analyzing files", unit="file"):
        filepath = os.path.join(directory, filename)
        try:
            img = nib.load(filepath)  # type: ignore
            data_types[img.get_data_dtype()] += 1  # type: ignore
            resolutions[tuple(img.shape)] += 1  # type: ignore
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

    print("Data types:")
    for dtype, count in data_types.items():
        print(f"  {dtype}: {count}")

    print("\nResolutions:")
    for resolution, count in resolutions.items():
        print(f"  {resolution}: {count}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script_name.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    analyze_nii_gz_files(directory)
