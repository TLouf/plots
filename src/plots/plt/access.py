from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def get_ax(fig: Figure, label: str):
    for ax in fig.axes:
        if ax.get_label() == label:
            return ax
    raise ValueError("no ax found with label {0}".format(label))


def get_colorbar_ax(fig: Figure):
    return get_ax(fig, "<colorbar>")
