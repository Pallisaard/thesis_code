import re
import subprocess


def split_except_single_quoted(string: str) -> list[str]:
    return re.split(r"\s+(?=(?:[^\']*\'[^\']*\')*[^\']*$)", string)


def execute_terminal_command(
    command: str, capture_output: bool = True, text: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    command_and_args = split_except_single_quoted(command)
    return subprocess.run(
        command_and_args, capture_output=capture_output, text=text, **kwargs
    )
