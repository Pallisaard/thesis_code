from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    Callback,
)


def get_checkpoint_callback(
    path: str, monitor: str, save_last: bool = True, save_top_k: int = 1
) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=path, monitor=monitor, save_last=save_last, save_top_k=save_top_k
    )


def get_summary_callback() -> RichModelSummary:
    return RichModelSummary()


def get_progress_bar_callback(
    refresh_rate: int = 1, leave: bool = False
) -> RichProgressBar:
    return RichProgressBar(refresh_rate=refresh_rate, leave=leave)


def get_standard_callbacks(checkpoint_path: str, monitor: str) -> list[Callback]:
    return [
        get_checkpoint_callback(path=checkpoint_path, monitor=monitor),
        get_summary_callback(),
        get_progress_bar_callback(),
    ]
