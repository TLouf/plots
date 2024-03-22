from __future__ import annotations

from typing import TYPE_CHECKING, Collection

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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


def dist_plot(data, base=10, bins=None, compl=False, cumul=False):
    if base < 1:
        raise ValueError("base should be >= 1")
    if bins is None:
        x_plot, y_plot = np.unique(data, return_counts=True)
    else:
        log_data = np.log(data) / np.log(base) if base > 1 else data
        y_plot, edges = np.histogram(log_data, bins=bins)
        mask = y_plot > 0
        x_plot = (edges[1:] + edges[:-1]) / 2
        if base > 1:
            x_plot = base**x_plot

    y_plot = y_plot / y_plot.sum()
    if cumul:
        y_plot = y_plot.cumsum()
    elif bins is not None:
        x_plot = x_plot[mask]
        y_plot = y_plot[mask]
    if compl:
        y_plot = 1 - y_plot
        if cumul:
            y_plot = y_plot[:-1]
            x_plot = x_plot[:-1]
    ylabel = f"{compl * 'C'}{cumul * 'C'}DF"
    return x_plot, y_plot, ylabel
