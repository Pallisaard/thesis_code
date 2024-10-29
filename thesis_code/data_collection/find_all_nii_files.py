import os
import argparse


def list_nii_gz_files(
    folder_path, replace_str=None, output_filename="nii_gz_files.txt"
):
    """
    Lists all .nii.gz files in a folder and writes their absolute paths to a file.

    Args:
      folder_path: Path to the folder containing the .nii.gz files.
      replace_str: Optional string in the format "before:after" to replace
                   "before" with "after" in the output paths.
    """

    with open(output_filename, "w") as f:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".nii.gz"):
                    abs_path = os.path.abspath(os.path.join(root, file))
                    if replace_str:
                        before, after = replace_str.split(":")
                        abs_path = abs_path.replace(before, after)
                    f.write(abs_path + "\n")

    print(f"Absolute paths of .nii.gz files written to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List .nii.gz files in a folder.")
    parser.add_argument(
        "folder_path", help="Path to the folder containing .nii.gz files"
    )
    parser.add_argument(
        "--replace",
        help="Optional string in the format 'before:after' to replace parts of the path",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="nii_gz_files.txt",
        help="Optional filename to write the output to. Default: nii_gz_files.txt",
    )
    args = parser.parse_args()

    list_nii_gz_files(args.folder_path, args.replace, args.output_name)
