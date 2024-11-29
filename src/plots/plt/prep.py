from __future__ import annotations

from typing import TYPE_CHECKING, Collection

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy import ndarray


def mosaic_from_dict(d, ncols, cax: str | None = None):
    mosaic = list(d.keys())
    nr_empty = len(mosaic) % ncols
    nrows = len(mosaic) // ncols + int(nr_empty > 0)
    mosaic.extend(["."] * nr_empty)
    mosaic = np.array(mosaic).reshape(nrows, ncols)
    if cax is not None:
        if cax.startswith("h"):
            mosaic = np.vstack([mosaic, np.array(["cax"] * ncols)])
        elif cax.startswith("v"):
            mosaic = np.hstack([mosaic, np.array([["cax"]] * nrows)])
        else:
            raise ValueError(f"`cax` must be 'h' or 'v' but '{cax}' was passed")
    return mosaic


def make_ax(ax: Axes | None = None, fig_kwargs: dict | None = None) -> Axes:
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        _, ax = plt.subplots()
    return ax


def make_axes(
    data_coll: Collection,
    axes: Axes | Collection[Axes] | None = None,
    cax: str | None = None,
    fig_kwargs: dict | None = None,
) -> Collection[Axes]:
    if fig_kwargs is None:
        fig_kwargs = {}
    nr_axes = len(data_coll)
    if axes is None:
        if isinstance(data_coll, dict):
            mosaic = mosaic_from_dict(data_coll, fig_kwargs["ncols"], cax)
            fig, axes = plt.subplot_mosaic(mosaic, **fig_kwargs)
        else:
            nrows = 1
            ncols = 1
            if cax is not None:
                nr_axes += 1
                if cax.startswith("h"):
                    nrows = 2
                elif cax.startswith("v"):
                    nrows = 2
            fig, axes = plt.subplots(nrows, ncols, **fig_kwargs)

    if nr_axes == 1:
        axes = (axes,)
    return axes


def dist_plot(
    x: str | ndarray | None = None,
    y: str | ndarray | None = None,
    data=None,
    base=10,
    bins=None,
    compl=False,
    cumul=False,
    density=False,
):
    if base < 1:
        raise ValueError("base should be >= 1")
    if density and bins is None:
        raise ValueError("cannot compute densities without binning")

    x = data[x] if isinstance(x, str) else x
    y = data[y] if isinstance(y, str) else y
    if bins is None:
        if x is None or y is None:
            x_plot, y_plot = np.unique(data, return_counts=True)
        else:
            x_plot, y_plot = np.asarray(x), np.asarray(y)
    else:
        if x is None:
            x = data
        x = np.asarray(x)
        if base > 1:
            bin_edges = np.logspace(
                np.log(x.min()) / np.log(base), np.log(x.max()) / np.log(base), bins + 1
            )
        else:
            bin_edges = np.linspace(x.min(), x.max(), bins + 1)

        if y is not None:
            y = np.asarray(y)

        y_plot, _ = np.histogram(x, bins=bin_edges, weights=y, density=density)
        if base > 1:
            x_plot = (bin_edges[1:] * bin_edges[:-1]) ** 0.5
        else:
            x_plot = (bin_edges[1:] + bin_edges[:-1]) / 2

    if density:
        ylabel = "PDF"
    else:
        y_plot = y_plot / y_plot.sum()
        if cumul:
            y_plot = y_plot.cumsum()
        if compl:
            y_plot = 1 - y_plot
            if cumul:
                y_plot = y_plot[:-1]
                x_plot = x_plot[:-1]
        ylabel = f"{compl * 'C'}{cumul * 'C'}DF"

    if bins is not None and base > 1:
        mask = y_plot > 0
        x_plot = x_plot[mask]
        y_plot = y_plot[mask]
    return x_plot, y_plot, ylabel


def binned_stat(
    x: str | np.ndarray,
    y: str | np.ndarray,
    data=None,
    bins=10,
    base=10,
    point_statistic="mean",
    error_percentile: int | None = None,
):
    if base < 1:
        raise ValueError("base should be >= 1")

    x = np.asarray(data[x] if isinstance(x, str) else x)
    y = np.asarray(data[y] if isinstance(y, str) else y)
    log_x = np.log(x) / np.log(base) if base > 1 else x
    binned_point_stat = scipy.stats.binned_statistic_dd(
        log_x, y, statistic=point_statistic, bins=bins
    )
    edges = binned_point_stat.bin_edges[0]
    y_plot = binned_point_stat.statistic
    mask = y_plot > 0
    bin_centers = (edges[1:] + edges[:-1]) / 2
    if base > 1:
        bin_centers = base**bin_centers

    if error_percentile is None:
        binned_std = scipy.stats.binned_statistic_dd(
            log_x,
            y,
            statistic="std",
            binned_statistic_result=binned_point_stat,
        )
        y_error = binned_std.statistic[mask]
        y_error = (y_error, y_error)
    else:
        half_compl_interval = (100 - error_percentile) / 2
        binned_lower_err = scipy.stats.binned_statistic_dd(
            log_x,
            y,
            statistic=lambda x: np.percentile(y, half_compl_interval),
            binned_statistic_result=binned_point_stat,
        )
        binned_upper_err = scipy.stats.binned_statistic_dd(
            log_x,
            y,
            statistic=lambda x: np.percentile(y, error_percentile + half_compl_interval),
            binned_statistic_result=binned_point_stat,
        )
        y_error = (binned_lower_err.statistic[mask], binned_upper_err.statistic[mask])

    bin_centers = bin_centers[mask]
    y_plot = y_plot[mask]
    return bin_centers, y_plot, y_error
