import subprocess
import os
import re
import tarfile
import tempfile
import argparse
from pathlib import Path
from typing import Any, Callable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse dataset arguments.")
    parser.add_argument(
        "--dataset-name", required=True, help="Name of the dataset (required)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
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
    return parser.parse_args()


def process_exported_data(
    tar_path: str, files_to_delete: list[str], output_path: Optional[str] = None
) -> None:
    output_path = output_path or tar_path.replace(".tar.gz", "_processed.tar.gz")
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        extracted_dir = next(Path(temp_dir).iterdir())
        delete_files_from_list(str(extracted_dir), files_to_delete)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(extracted_dir, arcname=os.path.basename(extracted_dir))


def delete_files_from_list(directory: str, file_list: list[str]) -> None:
    directory_path = Path(directory).resolve()
    for relative_path in file_list:
        (directory_path / relative_path).unlink(missing_ok=True)


def split_except_single_quoted(string: str) -> list[str]:
    return re.split(r"\s+(?=(?:[^\']*\'[^\']*\')*[^\']*$)", string)


def execute_terminal_command(
    command: str, capture_output: bool = True, text: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    command_and_args = split_except_single_quoted(command)
    return subprocess.run(
        command_and_args, capture_output=capture_output, text=text, **kwargs
    )


def find_files_with_substring(directory: str, substring: str) -> list[str]:
    return [
        str(file.relative_to(directory))
        for file in Path(directory).rglob("*")
        if substring in file.name and (file.is_file() or file.is_symlink())
    ]


def main() -> None:
    args = parse_args()

    os.chdir(args.data_path)
    print(f"Changed working directory to: {os.getcwd()}")

    print(f"installing dataset: {args.dataset_name}")
    install_commmand = (
        f"datalad install https://github.com/OpenNeuroDatasets/{args.dataset_name}.git"
    )
    execute_terminal_command(install_commmand)

    os.chdir(args.dataset_name)
    print(f"Changed working directory to: {os.getcwd()}")

    t1w_files = find_files_with_substring(".", "T1w")
    t1w_nii_files = [f for f in t1w_files if f.endswith(".nii.gz")]

    print(f"Number of .nii.gz files: {len(t1w_nii_files)}")

    for file in t1w_nii_files:
        print(f"Downloading: {file}")
        result = execute_terminal_command(f"datalad get {file}")
        print(f"Download result: {(result.stdout, result.returncode)}")

    os.chdir("..")
    print(f"Changed working directory to: {os.getcwd()}")

    export_command = f"datalad export-archive -d {args.dataset_name} --missing-content ignore {args.out_data_name}"
    print(f"Exporting data to: {args.out_data_name}")
    export_result = execute_terminal_command(export_command)
    print(f"Export result: {(export_result.stdout, export_result.returncode)}")

    drop_command = f"datalad drop --what filecontent -d {args.dataset_name}"
    print("Dropping old dataset")
    drop_result = execute_terminal_command(drop_command)
    print(f"Drop result: {(drop_result.stdout, drop_result.returncode)}")


if __name__ == "__main__":
    main()
