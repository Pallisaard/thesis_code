from pathlib import Path
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    Callback,
)


def get_checkpoint_callback(
    path: str | Path,
    model_name: str,
    filename: str | None = None,
    monitor: str = "val_total_loss",
    save_last: bool = True,
    save_top_k: int = 1,
) -> ModelCheckpoint:
    if filename is None:
        filename = "{epoch}-{step}-{val_total_loss:.4f}"

    if isinstance(path, str):
        path = Path(path)

    path = path / model_name
    # if path exists, add "_n" to the end of the path. Let n start from 1 and go up until we find a path that doesn't exist
    if not path.exists():
        path.mkdir(parents=True)
    else:
        n = 1
        while path.exists():
            path = path.with_name(f"{path.name}_{n}")
            n += 1

    return ModelCheckpoint(
        dirpath=path,
        monitor=monitor,
        save_last=save_last,
        save_top_k=save_top_k,
        filename=filename,
    )


DEFAULT_CHECKPOINT_PATH = "lightning/checkpoints"


def get_summary_callback() -> RichModelSummary:
    return RichModelSummary()


def get_progress_bar_callback(
    refresh_rate: int = 1, leave: bool = False
) -> RichProgressBar:
    return RichProgressBar(refresh_rate=refresh_rate, leave=leave)


def get_standard_callbacks(
    checkpoint_path: str, model_name: str, monitor: str
) -> list[Callback]:
    return [
        get_checkpoint_callback(
            path=checkpoint_path, model_name=model_name, monitor=monitor
        ),
        get_summary_callback(),
        get_progress_bar_callback(),
    ]
