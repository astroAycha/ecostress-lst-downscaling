from typing import Dict
import numpy as np
from rasterio.warp import transform_bounds
from affine import Affine
from scipy.ndimage import distance_transform_edt
import xarray as xr
from preprocess_lst import qc_mask_lst, clip_and_mask_lst


def build_reference_grid(aoi: tuple, 
                         target_crs: str, 
                         resolution: float) -> tuple[Affine, tuple[int, int]]:
    """
    Build a target transform + shape spanning the full AOI,
    (independent of any individual tile's footprint)

    Parameters
    ----------
    aoi: tuple
        tuple of (west, south, east, north) in EPSG:4326
    target_crs: str
        target coordinate reference system for reprojection
    resolution: float
        desired pixel size in target CRS units (e.g. meters)

    Returns
    -------
     transform: 
        affine transform for the target grid
     shape: 
        tuple of (height, width) for the target grid
    """
    west, south, east, north = aoi
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", 
                                              target_crs, 
                                              west, south, east, north)

    width  = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)

    return transform, (height, width)

####################

def build_composite(entries: list[Dict], 
                    aoi: tuple, 
                    target_crs: str, 
                    transform: Affine, 
                    shape: tuple[int, int],
                    max_cloud_cover: int) -> xr.DataArray | None:
    """
    Build a composite LST DataArray for a single tile by:
    1. Applying QC, cloud and water masks to each granule's LST
    2. Reprojecting each masked LST to a common grid defined by the 
    AOI and target CRS
    3. Averaging the reprojected LSTs across time to create 
    a composite for that tile.
    
    Reprojects every date onto the SAME shared grid (built once from the
    AOI, not from any individual granule) before averaging, so a date with
    partial tile coverage can't shrink the result.

    Parameters
    ----------
    entries: list[Dict]
        list of {'lst_file', 'water_file', 'qc_file', 'cloud_file'} 
        for a given tile, across multiple acquisition dates.
    aoi: tuple
        tuple of (west, south, east, north) defining the area of interest
    target_crs: str
        target coordinate reference system for reprojection
    transform: Affine
        affine transform for the target grid
    shape: tuple[int, int]
        tuple of (height, width) for the target grid
    max_cloud_cover: int
        maximum allowed cloud cover fraction (0-100) for a granule to be 
        included

    Returns
    -------
    xr.DataArray or None
        Composite LST for the tile, or None if no good pixels 
        remain after masking.
    """
    matched_arrays = []
    granule_acq_dates = []

    for entry in entries:
        # apply QC mask
        lst_masked = qc_mask_lst(entry['lst_file'], entry['qc_file'])
        if lst_masked is None:
            continue

        # apply water, cloud masks and clipping to AOI
        lst_clipped = clip_and_mask_lst(lst_masked, 
                                    entry['water_file'], 
                                    entry['cloud_file'], 
                                    aoi, 
                                    target_crs=target_crs,
                                    max_cloud_cover=max_cloud_cover)
        if lst_clipped is None:
            continue

        lst_matched = lst_clipped.rio.reproject(target_crs, 
                                                shape=shape,
                                                nodata=np.nan,
                                                transform=transform)
        # re-apply nodata mask after reproject to catch any bleed-through
        lst_matched = lst_matched.where(lst_matched != np.nan)
        lst_matched = lst_matched.rio.write_nodata(np.nan)
        temp_extent = entry['granule']['umm']['TemporalExtent']
        acquisition_date = temp_extent['RangeDateTime']['BeginningDateTime']
        granule_acq_dates.append(acquisition_date)
        matched_arrays.append(lst_matched)

    if not matched_arrays:
        print(" >>> No usable granules for this tile after QC, water, and cloud masking ❗")
        return None

    print(" --->>> Building composite from the following acquisition dates:")
    print(f"{granule_acq_dates}")
    stacked = xr.concat(matched_arrays, dim='time')
    composite = stacked.mean(dim='time', skipna=True)
    composite = composite.rio.write_crs(target_crs).rio.write_nodata(np.nan)

    return composite

########################

def edge_distance_weight(da: xr.DataArray, 
                         feather_px: int = 15) -> xr.DataArray:
    """
    Weight map that ramps from 0 at the edge of valid data up to 1
    after `feather_px` pixels inward. Used to feather tile boundaries.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with valid and invalid pixels (NaN)
    feather_px : int
        Number of pixels over which to feather the edge (default: 15)

    Returns
    -------
    xr.DataArray
        Weight map with values in [0, 1], same shape and coords as input.
    
    """
    valid = da.notnull().values
    dist = distance_transform_edt(valid) # pixels to nearest invalid/edge pixel
    weight = np.clip(dist / feather_px, 0, 1) # ramp 0 to 1 over feather_px pixels
    return xr.DataArray(weight, coords=da.coords, dims=da.dims)


##############################

def merge_tiles(tile_composites: dict[str, xr.DataArray],
                 target_crs: str, 
                 feather_px: int = 15) -> xr.DataArray:
    """
    Feather-blend multiple tile composites into a single mosaic.

    Parameters
    ----------
    tile_composites: dict
        dict of {tile_id: xr.DataArray} for each tile's composite LST

    target_crs: str
        target coordinate reference system for reprojection
    
    feather_px: int
        number of pixels over which to feather tile edges (default 15)

    Returns
    -------
    xr.DataArray
        Feather-blended mosaic of all tiles.
    """
    arrays = list(tile_composites.values())
    weights = [edge_distance_weight(da, feather_px) for da in arrays]

    stacked_data = xr.concat(arrays, dim='tile')
    stacked_weights = xr.concat(weights, dim='tile').where(stacked_data.notnull(), 0)

    weighted_sum = (stacked_data.fillna(0) * stacked_weights).sum(dim='tile')
    weight_total = stacked_weights.sum(dim='tile', skipna=True)

    mosaic = (weighted_sum / weight_total).where(weight_total > 0)
    mosaic = mosaic.rio.write_crs(target_crs).rio.write_nodata(np.nan)

    return mosaic