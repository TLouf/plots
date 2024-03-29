import numpy as np

from plots.dependencies import geo as geo_utils
from plots.dependencies import plotly as plotly


def calc_fit_zoom(gdf):
    max_bound = max(geo_utils.calc_shape_dims(gdf))
    zoom = 11.5 - np.log(max_bound)
    return zoom


def plot_interactive(
    fig,
    mapbox_style="stamen-toner",
    mapbox_zoom=10,
    plotly_renderer="iframe_connected",
    save_path=None,
    show=False,
):
    """
    Utility to plot an interactive map, given the plot data and layout given in
    the plotly Figure instance `fig`.
    """
    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox_zoom=mapbox_zoom,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    if save_path:
        # Path objects are not yet supported by plotly, so first cast to str.
        plotly.offline.plot(fig, filename=str(save_path), auto_open=False)
    if show:
        fig.show(
            renderer=plotly_renderer,
            width=900,
            height=600,
            config={"modeBarButtonsToAdd": ["zoomInMapbox", "zoomOutMapbox"]},
        )

    return fig


def cells(
    gdf,
    metric_col,
    colorscale="Plasma",
    latlon_proj="epsg:4326",
    alpha=0.8,
    **plot_interactive_kwargs,
):
    """
    Plots an interactive Choropleth map with Plotly.
    The map layer on top of which this data is shown is provided by mapbox (see
    https://plot.ly/python/mapbox-layers/#base-maps-in-layoutmapboxstyle for
    possible values of 'mapbox_style').
    Plotly proposes different renderers, described at:
    https://plot.ly/python/renderers/#the-builtin-renderers.
    The geometry column of gdf must contain only valid geometries:
    just one null value will prevent the choropleth from being plotted.
    """
    latlon_gdf = gdf["geometry"].to_crs(latlon_proj)
    plot_interactive_kwargs.setdefault("mapbox_zoom", calc_fit_zoom(latlon_gdf))

    minx, miny, maxx, maxy = gdf.total_bounds
    layout = plotly.graph_objs.Layout(
        mapbox_center={"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}
    )

    # Get a dictionary corresponding to the geojson (because even though the
    # argument is called geojson, it requires a dict type, not a str). The
    # geometry must be in lat, lon.
    geo_dict = latlon_gdf.geometry.__geo_interface__
    choropleth_dict = dict(
        geojson=geo_dict,
        locations=gdf.index.values,
        colorscale=colorscale,
        marker_opacity=alpha,
        marker_line_width=0.1,
        z=gdf[metric_col],
        visible=True,
    )

    data = [plotly.graph_objs.Choroplethmapbox(**choropleth_dict)]

    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    fig = plot_interactive(fig, **plot_interactive_kwargs)
    return fig
