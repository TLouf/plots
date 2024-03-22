from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plots.plt.colors as colors
from plots.dependencies import geopandas as geopd
from plots.dependencies import shapely

from .colors import get_norm
from .prep import make_axes

if TYPE_CHECKING:
    from typing import Collection

    import pandas as pd
    from geo.region import BaseRegion


def discrete_choro(
    data: pd.Series,
    colors_dict: dict,
    regions: Collection[BaseRegion],
    fig_kwargs: dict | None = None,
    show_lgd=True,
    fig=None,
    axes=None,
    **kwargs,
):
    """
    Plot a clustering level, with cells of the regions coloured according to the cluster
    they belong to, shown in a legend. They are not drawn (left in white/transparent) if
    no information is available on their belonging to a cluster.
    """
    axes = make_axes(regions, axes, fig_kwargs=fig_kwargs)

    for ax, reg in zip(axes, regions):
        cc_geodf = reg.cells_geodf.join(data, how="inner")

        for label, label_geodf in cc_geodf.groupby("labels"):
            # Don't put a cmap in kwargs['plot'] because here we use a
            # fixed color per cluster.
            label_geodf.plot(ax=ax, color=colors_dict[label], **kwargs.get("plot", {}))

        reg.shape_geodf.plot(
            ax=ax,
            color="none",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(reg.readable)
        ax.set_axis_off()

    fig = ax.get_figure()

    if show_lgd:
        lgd_container = ax if len(regions) == 1 else fig
        lgd_kwargs = {**{"loc": "center right"}, **kwargs.get("legend", {})}
        # The colours will correspond because groupby sorts by the column by
        # which we group, and we sorted the unique labels.
        _ = colors.colored_poly_legend(lgd_container, colors_dict, **lgd_kwargs)

    return fig, axes


def continuous_choro(
    data: pd.Series,
    regions: Collection[BaseRegion],
    axes=None,
    cax=None,
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    vcenter=None,
    null_color="gray",
    cbar_label=None,
    cbar_kwargs=None,
    normed_bboxes=None,
    area_from_cells_union=False,
    clip_to_cells=False,
    fig_kwargs: dict | None = None,
    **plot_kwargs,
):
    """
    Make a choropleth map from continuous values given in `data` for some given
    regions. A colorbar will be drawn, either in the given `cax` or to the right of the
    last ax in `axes`. Cells missing in `data` are coloured in `null_color`.
    """
    if cbar_kwargs is None:
        cbar_kwargs = {}

    axes = make_axes(regions, axes, fig_kwargs=fig_kwargs)

    norm = get_norm(data, norm, vmin, vmax, vcenter)

    for ax, reg in zip(axes, regions):
        plot_df = reg.cells_geodf.rename_axis("cell_id").join(data, how="inner")
        plot_area = plot_df.shape[0] > 0.5 * reg.cells_geodf.shape[0]
        if plot_area:
            area_gdf = reg.shape_geodf.copy()
            if clip_to_cells:
                area_gdf = area_gdf.clip(shapely.geometry.box(*plot_df.total_bounds))
            elif area_from_cells_union:
                union = plot_df.unary_union
                union = shapely.geometry.MultiPolygon(
                    [poly for poly in union.geoms if poly.area > union.area / 1000]
                ).simplify(100)
                area_gdf = geopd.GeoSeries(union, crs=plot_df.crs)
        if plot_area:
            area_gdf.plot(ax=ax, color=null_color, edgecolor="none", alpha=0.3)

        plot_df.plot(
            column=data.name, ax=ax, norm=norm, cmap=cmap, **plot_kwargs
        )

        if plot_area:
            area_gdf.plot(ax=ax, color="none", edgecolor="black", linewidth=0.5)
        if len(regions) > 1:
            ax.set_title(reg.readable)
        ax.set_axis_off()

    fig = ax.get_figure()

    if cbar_label is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if cax is None:
            divider = make_axes_locatable(ax)
            # Create an axes on the right side of ax. The width of cax will be 5% of ax
            # and the padding between cax and ax will be fixed at 0.1 inch.
            cax_pos = (
                "bottom" if cbar_kwargs.get("orientation") == "horizontal" else "right"
            )
            cax = divider.append_axes(cax_pos, size="5%", pad=0.1)
        axes = np.append(axes, cax)

        _ = fig.colorbar(sm, cax=cax, label=cbar_label, **cbar_kwargs)

    if normed_bboxes is not None:
        for ax, bbox in zip(axes, normed_bboxes):
            ax.set_position(bbox)

    return fig, axes
