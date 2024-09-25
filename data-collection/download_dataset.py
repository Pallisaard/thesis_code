import subprocess
import os
import re
import tarfile
import tempfile
import argparse
from argparse import Namespace
from collections.abc import Callable
from pprint import pprint
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Parse dataset arguments.")

    # Add arguments
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="id for the datalad dataset in format dsXXXXXX (required)",
    )
    parser.add_argument(
        "--data-path",
        default=".",
        help="Path to the data directory (default: current directory)",
    )
    parser.add_argument(
        "--out-data-name",
        default="out_data",
        help="Name of the output data (default: out_data)",
    )
    parser.add_argument(
        "--file-count-limit",
        type=int,
        default=None,
        help="Limit the number of t1w_files via indexing (default: no limit)",
    )

    # Parse arguments
    args = parser.parse_args()

    return args


def process_exported_data(
    tar_path: str, files_to_delete: list[str], output_path: str | None = None
):
    if output_path is None:
        output_path = tar_path.replace(".tar.gz", "_processed.tar.gz")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the tar.gz file
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Find the extracted directory (assuming it's the only directory in temp_dir)
        extracted_dir = next(Path(temp_dir).iterdir())

        # Delete files
        delete_files_from_list(str(extracted_dir), files_to_delete)

        # Create a new tar.gz file with the remaining contents
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(extracted_dir, arcname=os.path.basename(extracted_dir))


# The delete_files_from_list function (unchanged from previous example)
def delete_files_from_list(directory: str, file_list: list[str]):
    directory_path = Path(directory).resolve()

    for relative_path in file_list:
        full_path = directory_path / relative_path
        os.remove(full_path)


def create_logger(to_stdout: bool):
    def log(log_val: Any):
        if to_stdout:
            print(log_val) if isinstance(log_val, str) else pprint(log_val)

    return log


def split_except_single_quoted(string: str) -> list[str]:
    # The pattern splits on all spaces except for those inside single quotes.
    # Provided by ChatGPT. LGTM.
    pattern = r"\s+(?=(?:[^\']*\'[^\']*\')*[^\']*$)"
    return re.split(pattern, string)


def execute_persistant_chdir(change_path: str):
    os.chdir(change_path)
    return os.getcwd()


def execute_zsh_command(
    command: str, capture_output: bool = True, text: bool = True, **kwargs
) -> CompletedProcess[str]:
    command_and_args = split_except_single_quoted(command)
    result = subprocess.run(
        command_and_args, capture_output=capture_output, text=text, **kwargs
    )
    return result


def find_file_substring_cmp(
    directory: str, substring: str, cmp: Callable[[str, str], bool]
):
    matching_files = []
    directory_path = Path(directory)

    for file_path in directory_path.rglob("*"):
        if (file_path.is_file() or file_path.is_symlink()) and cmp(
            substring, file_path.name
        ):
            relative_path = file_path.relative_to(directory_path)
            matching_files.append(str(relative_path))

    return matching_files


def find_files_with_substring(directory: str, substring: str):
    return find_file_substring_cmp(directory, substring, lambda x, y: x in y)


def find_files_without_substring(directory: str, substring: str):
    return find_file_substring_cmp(directory, substring, lambda x, y: x not in y)


# data_path = "../data"
# dataset_name = "ds003653"
# out_data_name = "test_ds"
MRI_TYPE = "T1w"
LOG_TO_STDOUT = True


def main() -> None:
    args = parse_args()
    log = create_logger(to_stdout=True)

    os.chdir(args.data_path)
    log(f"Changed working directory to: {os.getcwd()}")

    result = execute_zsh_command(
        f"datalad install https://github.com/OpenNeuroDatasets/{args.dataset_id}.git"
    )

    t1w_files = find_files_with_substring(args.dataset_id, "T1w")
    t1w_nii_files = [f for f in t1w_files if f.endswith(".nii.gz")]

    log(f"Number of .nii.gz files: {len(t1w_nii_files)}")

    os.chdir(args.dataset_id)
    log(f"Changed working directory to: {os.getcwd()}")

    for file in t1w_files:
        log(f"- calling datalad get command for '{file}'")
        datalad_get_command = f"datalad get {file}"
        get_result = execute_zsh_command(datalad_get_command)
        # log(f"- get result return code: '{get_result.returncode}'")
        log(f"  - {(get_result.stdout, get_result.returncode)}")

    os.chdir("..")
    log(f"Changed working directory to: {os.getcwd()}")

    export_command = f"datalad export-archive -d {args.dataset_id} --missing-content ignore {args.out_data_name}"
    log(f"Exporting data to: {args.out_data_name}")
    export_result = execute_zsh_command(export_command)
    log(f"Export result: {(export_result.stdout, export_result.returncode)}")

    drop_command = f"datalad drop --what filecontent -d {args.dataset_id}"
    log("Dropping old dataset")
    drop_result = execute_zsh_command(drop_command)
    log(f"Drop result: {(drop_result.stdout, drop_result.returncode)}")


if __name__ == "__main__":
    main()
