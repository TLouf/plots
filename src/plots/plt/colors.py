import colorsys

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


def gen_distinct_colors(num_colors, mean_lightness=40):
    """
    Generate `num_colors` distinct colors for a discrete colormap, in the format
    of a list of tuples of normalized RGB values.
    """
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (mean_lightness + np.random.rand() * 20) / 100.0
        saturation = (80 + np.random.rand() * 20) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def colored_poly_legend(container, label_color, **lgd_kwargs):
    """
    Adds to a legend with colored points to a `container`, which can be a plt ax
    or figure. The color of the points and their associated labels are given
    respectively as values and keys of the dict `label_color`.
    """
    handles = [mpatches.Patch(color=c, label=l) for l, c in label_color.items()]
    kwargs = {**{"handlelength": 1, "handleheight": 1}, **lgd_kwargs}
    container.legend(handles=handles, **kwargs)
    return container


def get_norm(
    plot_series,
    norm=None,
    vmin=None,
    vmax=None,
    vcenter=None,
) -> mcolors.Normalize:

    if norm is None:
        if vmin is None:
            vmin = plot_series.min()
        if vmax is None:
            vmax = plot_series.max()
        if vcenter is None:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)

    if vmin is not None:
        norm.vmin = vmin
    if vmax is not None:
        norm.vmax = vmax
    return norm
