import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Literal, Optional, List, Tuple, Union
import copy


def plot_loss_curves(
    df: pd.DataFrame,
    x_column: str = "step",
    value_columns: Optional[List[str]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = False,
    style: str = "whitegrid",
    alpha: float = 0.8,
    linewidth: float = 2,
) -> plt.Figure:
    """
    Plot multiple loss curves from a DataFrame using seaborn.

    Args:
        df (pd.DataFrame): DataFrame containing the loss values and steps
        x_column (str): Name of the column containing x-axis values (default: 'step')
        value_columns (List[str], optional): List of column names to plot. If None, plots all columns except x_column
        title (str): Title of the plot
        figsize (Tuple[int, int]): Size of the figure (width, height)
        log_scale (bool): Whether to use log scale for y-axis
        style (str): Style of the plot (seaborn style)
        alpha (float): Transparency of the lines
        linewidth (float): Width of the lines

    Returns:
        plt.Figure: The generated figure object
    """
    # Set the style
    sns.set_style(style)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # If no value columns specified, use all columns except x_column
    if value_columns is None:
        value_columns = [col for col in df.columns if col != x_column]

    # Plot each column using the current color cycle
    for column in value_columns:
        sns.lineplot(
            data=df,
            x=x_column,
            y=column,
            label=column,
            alpha=alpha,
            linewidth=linewidth,
        )

    # Customize the plot
    plt.title(title, pad=20)
    plt.xlabel(x_column.capitalize())
    plt.ylabel("Loss")

    if log_scale:
        plt.yscale("log")

    # Customize grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Customize legend
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    return fig


def create_plot_grid(
    figs: List[plt.Figure],
    grid_size: Tuple[int, int],
    super_title: str = "",
    figsize: Optional[Tuple[int, int]] = None,
    spacing: float = 0.3,
) -> plt.Figure:
    """
    Arrange multiple figures in a grid layout.

    Args:
        figs (List[plt.Figure]): List of figures to arrange in grid
        grid_size (Tuple[int, int]): Size of grid as (width, height)
        super_title (str): Title for the entire figure
        figsize (Tuple[int, int], optional): Size of the combined figure.
            If None, automatically calculated based on input figures
        spacing (float): Spacing between subplots as a fraction of figure size

    Returns:
        plt.Figure: Combined figure with grid of plots
    """
    width, height = grid_size
    n_plots = width * height

    if len(figs) != n_plots:
        raise ValueError(
            f"Number of figures ({len(figs)}) must match grid size ({n_plots})"
        )

    # Calculate figsize if not provided
    if figsize is None:
        # Get size of first figure as reference
        ref_fig = figs[0]
        ref_width, ref_height = ref_fig.get_size_inches()

        # Scale up based on grid size and add padding
        total_width = ref_width * width * (1 + spacing)
        total_height = ref_height * height * (1 + spacing)
        if super_title:
            total_height += 1  # Extra space for super title
        figsize = (total_width, total_height)

    # Create new figure
    fig = plt.figure(figsize=figsize)

    # Add super title if provided
    if super_title:
        fig.suptitle(super_title, fontsize=16, y=0.95)

    # Create grid of subplots
    for idx, src_fig in enumerate(figs):
        # Create subplot
        ax = fig.add_subplot(height, width, idx + 1)

        # Copy contents from source figure
        src_ax = src_fig.axes[0]

        # Copy all artists (lines, texts, etc.) from source axis to new axis
        for artist in src_ax.get_children():
            if isinstance(artist, (plt.Line2D, plt.Text)):
                artist_copy = copy.copy(artist)
                ax.add_artist(artist_copy)

        # Copy axis properties
        ax.set_xlabel(src_ax.get_xlabel())
        ax.set_ylabel(src_ax.get_ylabel())
        ax.set_title(src_ax.get_title())
        ax.set_xlim(src_ax.get_xlim())
        ax.set_ylim(src_ax.get_ylim())

        # Copy grid
        ax.grid(src_ax.get_grid())

        # Copy scale (log/linear)
        ax.set_xscale(src_ax.get_xscale())
        ax.set_yscale(src_ax.get_yscale())

        # Handle legend
        if src_ax.get_legend() is not None:
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
                frameon=True,
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig


def plot_mri_slice(
    volume: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    target_size: Union[Literal[64], Literal[256]] = 256,
    cmap: str = "gray",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot a slice from a 3D MRI volume.

    Args:
        volume (np.ndarray): 3D numpy array containing the MRI volume
        slice_idx (int, optional): Index of slice to plot. If None, uses middle slice
        axis (int): Axis along which to take the slice (0, 1, or 2)
        target_size (int): Target size for the slice, must be either 64 or 256
        cmap (str): Colormap to use for plotting
        title (str, optional): Title for the plot
        figsize (Tuple[int, int]): Size of the figure

    Returns:
        plt.Figure: The generated figure object
    """
    # Input validation
    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D")

    if target_size not in [64, 256]:
        raise ValueError("target_size must be either 64 or 256")

    if axis not in [0, 1, 2]:
        raise ValueError("axis must be 0, 1, or 2")

    # Get slice
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2

    if axis == 0:
        slice_data = volume[slice_idx, :, :]
    elif axis == 1:
        slice_data = volume[:, slice_idx, :]
    else:  # axis == 2
        slice_data = volume[:, :, slice_idx]

    # Assert correct size
    assert slice_data.shape == (target_size, target_size), (
        f"Slice shape {slice_data.shape} does not match target size ({target_size}, {target_size})"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot slice
    im = ax.imshow(slice_data, cmap=cmap, aspect="equal", interpolation="nearest")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add title if provided
    if title:
        plt.title(title)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    return fig


def plot_mri_slices_grid(
    volumes: List[np.ndarray],
    grid_size: Tuple[int, int],
    slice_idx: Optional[int] = None,
    axis: int = 2,
    target_size: Union[Literal[64], Literal[256]] = 256,
    super_title: str = "",
    cmap: str = "gray",
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot multiple MRI slices in a grid layout.

    Args:
        volumes (List[np.ndarray]): List of 3D MRI volumes
        grid_size (Tuple[int, int]): Size of grid as (width, height)
        slice_idx (int, optional): Index of slice to plot. If None, uses middle slice
        axis (int): Axis along which to take the slice (0, 1, or 2)
        target_size (int): Target size for each slice, must be either 64 or 256
        super_title (str): Title for the entire figure
        cmap (str): Colormap to use for plotting
        figsize (Tuple[int, int], optional): Size of the figure. If None, calculated automatically

    Returns:
        plt.Figure: The generated figure object
    """
    # Create individual plots
    figs = []
    for volume in volumes:
        fig = plot_mri_slice(
            volume,
            slice_idx=slice_idx,
            axis=axis,
            target_size=target_size,
            cmap=cmap,
        )
        figs.append(fig)
        plt.close(fig)  # Close individual figures to save memory

    # Create grid
    grid_fig = create_plot_grid(
        figs,
        grid_size=grid_size,
        super_title=super_title,
        figsize=figsize,
    )

    return grid_fig


def plot_mri_slices_line(
    volumes: List[np.ndarray],
    labels: Optional[List[str]] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    target_size: Union[Literal[64], Literal[256]] = 256,
    cmap: str = "gray",
    show_arrow: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    spacing: float = 0.1,
) -> plt.Figure:
    """
    Plot multiple MRI slices in a horizontal line with optional labels and arrow.

    Args:
        volumes (List[np.ndarray]): List of 3D MRI volumes
        labels (List[str], optional): Labels for each volume to display below
        slice_idx (int, optional): Index of slice to plot. If None, uses middle slice
        axis (int): Axis along which to take the slice (0, 1, or 2)
        target_size (int): Target size for each slice, must be either 64 or 256
        cmap (str): Colormap to use for plotting
        show_arrow (bool): Whether to draw an arrow connecting first and last image
        figsize (Tuple[int, int], optional): Size of the figure. If None, calculated automatically
        spacing (float): Spacing between subplots as a fraction of subplot width

    Returns:
        plt.Figure: The generated figure object
    """
    n_plots = len(volumes)

    # Calculate figure size if not provided
    if figsize is None:
        base_size = 8 if target_size == 256 else 4
        width = base_size * n_plots * (1 + spacing)
        height = base_size * (1.2 if labels or show_arrow else 1)
        figsize = (width, height)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Calculate subplot positions
    width_ratio = 1 / (n_plots + (n_plots - 1) * spacing)

    # Create subplots
    axes = []
    for i in range(n_plots):
        # Calculate position for this subplot
        left = i * width_ratio * (1 + spacing)
        bottom = 0.15 if (labels or show_arrow) else 0.1
        width = width_ratio
        height = 0.8 if (labels or show_arrow) else 0.85

        ax = fig.add_axes([left, bottom, width, height])
        axes.append(ax)

        # Get and plot slice
        if axis == 0:
            slice_data = volumes[i][slice_idx or volumes[i].shape[axis] // 2, :, :]
        elif axis == 1:
            slice_data = volumes[i][:, slice_idx or volumes[i].shape[axis] // 2, :]
        else:  # axis == 2
            slice_data = volumes[i][:, :, slice_idx or volumes[i].shape[axis] // 2]

        assert slice_data.shape == (target_size, target_size), (
            f"Slice shape {slice_data.shape} does not match target size ({target_size}, {target_size})"
        )

        im = ax.imshow(slice_data, cmap=cmap, aspect="equal", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add label if provided
        if labels:
            ax.set_xlabel(labels[i], fontsize=12, labelpad=10)

    # Add arrow if requested
    if show_arrow:
        first_ax = axes[0]
        last_ax = axes[-1]

        # Get positions for arrow
        start_x = first_ax.get_position().x0
        end_x = last_ax.get_position().x1
        arrow_y = 0.05  # Position from bottom

        # Draw arrow
        plt.arrow(
            start_x,
            arrow_y,
            end_x - start_x,
            0,
            transform=fig.transFigure,
            head_width=0.02,
            head_length=0.02,
            length_includes_head=True,
            color="black",
            overhang=0.2,
        )

    return fig
