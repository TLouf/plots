from __future__ import annotations

from .prep import make_ax


def loglog(
    x_plot,
    y_plot,
    base=10,
    xlabel=None,
    ylabel=None,
    ax=None,
    grid_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
    **scatter_kwargs,
):
    if base < 1:
        raise ValueError("base should be >= 1")
    ax = make_ax(ax, fig_kwargs=fig_kwargs)
    ax.scatter(x_plot, y_plot, **scatter_kwargs)
    if base > 1:
        ax.set_xscale("log", base=base)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if grid_kwargs:
        ax.grid(**grid_kwargs)
    return ax
