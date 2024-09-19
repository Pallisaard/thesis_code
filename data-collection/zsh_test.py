from collections.abc import Callable
import subprocess
import os
import re
from pprint import pprint
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import tarfile
import tempfile


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


DATA_PATH = "../data"
DATASET_NAME = "ds003653"
MRI_TYPE = "T1w"
OUT_DATA_NAME = "test_ds"
LOG_TO_STDOUT = True


def main() -> None:
    log = create_logger(to_stdout=LOG_TO_STDOUT)

    # Change to
    execute_persistant_chdir(DATA_PATH)

    # Finding all T1w files.
    log("- gathering t1w files.")
    t1w_files = find_files_with_substring(DATASET_NAME, MRI_TYPE)
    # t1w_files = t1w_files[:5]
    log("- T1w files:")
    log(t1w_files)

    t1w_nii_files = [x for x in t1w_files if ".nii.gz" in x]
    print(f"- number of nii files: {len(t1w_nii_files)}")

    # downloading all T1w files.
    log(f"- calling: cd {DATASET_NAME}")
    execute_persistant_chdir(DATASET_NAME)
    log("- change dir successful")

    for file in t1w_files:
        log(f"- calling datalad get command for '{file}'")
        datalad_get_command = f"datalad get {file}"
        get_result = execute_zsh_command(datalad_get_command)
        log(f"- get result return code: '{get_result.returncode}'")
        # log("  - " + (get_result.stdout, get_result.returncode))

    log("- calling: cd ..")
    execute_persistant_chdir("..")
    log("- change dir successful")

    # Export downloads to tar.gz
    datalad_export_command = (
        f"datalad export-archive -d {DATASET_NAME}"
        + f" --missing-content ignore {OUT_DATA_NAME}"
    )
    log(f"exporting data to '{OUT_DATA_NAME}'")
    export_results = execute_zsh_command(datalad_export_command)
    log("- export results")
    log((export_results.stdout, export_results.returncode))

    # Drop datalad dataset
    datalad_drop_command = f"datalad drop --what filecontent -d {DATASET_NAME}"
    log("- dropping old dataset")
    drop_results = execute_zsh_command(datalad_drop_command)
    log("- drop results:")
    log((drop_results.stdout, drop_results.returncode))


if __name__ == "__main__":
    main()
