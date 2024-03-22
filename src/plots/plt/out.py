from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def output(
    ax: Axes | None = None,
    fig: Figure | None = None,
    show: bool = True,
    save_path: Path | None = None,
    **save_kwargs,
):
    if fig is None:
        fig = ax.get_figure()
    if show:
        fig.show()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, **save_kwargs)
