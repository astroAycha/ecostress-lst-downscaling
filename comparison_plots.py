"""Script to generate comparison plots for 70m and 10m LST data."""

import matplotlib.pyplot as plt
import hvplot.xarray
from bokeh.models import NumeralTickFormatter
import rioxarray as rxr
import numpy as np


def plot_comparison(path_70m: str, 
                    path_10m: str, 
                    aoi: str,
                    date_time: str):
    """
    Generate comparison plots for 70m and 10m LST data.

    Parameters
    ----------
    path_70m: str
        path to 70m LST data
    path_10m: str
        path to 10m LST data
    aoi: str
        area of interest
    date_time: str
        date and time of the data
    """

    lst_70m= rxr.open_rasterio(path_70m, masked=True).squeeze()
    lst_10m= rxr.open_rasterio(path_10m, masked=True).squeeze()

    vals = np.concatenate([
        lst_70m.values.ravel(),
        lst_10m.values.ravel()
    ])

    vals = vals[~np.isnan(vals)]

    vmin = np.percentile(vals, 2)
    vmax = np.percentile(vals, 98)

    cmap_reversed = plt.cm.RdYlBu_r

    plot_70m = lst_70m.rio.reproject("EPSG:4326").hvplot.image(
        x='x', y='y',
        cmap=cmap_reversed,
        clim=(vmin, vmax),
        title=f'{aoi} - {date_time} LST (70 m)', width=400, height=340,
        xlabel='Easting (m)', ylabel='Northing (m)'
    ).opts(xrotation=45,
        xformatter=NumeralTickFormatter(format='0,0'), 
        yformatter=NumeralTickFormatter(format='0,0'))

    plot_10m = lst_10m.rio.reproject("EPSG:4326").hvplot.image(
        x='x', y='y',
        cmap=cmap_reversed,
        clim=(vmin, vmax),
        title=f'{aoi} - {date_time} LST (10 m)', width=400, height=340,
        xlabel='Easting (m)', ylabel=''
    ).opts(xrotation=45,
        xformatter=NumeralTickFormatter(format='0,0'),
        yformatter=NumeralTickFormatter(format='0,0'))

    return plot_70m + plot_10m