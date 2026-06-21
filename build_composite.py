import numpy as np
from rasterio.warp import transform_bounds
from affine import Affine
from scipy.ndimage import distance_transform_edt
import xarray as xr
from preprocess_lst import qc_mask_lst, clip_and_mask_lst


def build_reference_grid(aoi, target_crs, resolution):
    """
    Build a target transform + shape spanning the full AOI,
    (independent of any individual tile's footprint)

    Parameters
    ----------
    aoi: 
        tuple of (west, south, east, north) in EPSG:4326
    target_crs: 
        target coordinate reference system for reprojection
    resolution: 
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

def build_composite(entries, aoi, target_crs, transform, shape):
    """
    Build a composite LST DataArray for a single tile by:
    1. Applying QC and water masks to each granule's LST
    2. Reprojecting each masked LST to a common grid defined by the 
    AOI and target CRS
    3. Averaging the reprojected LSTs across time to create 
    a composite for that tile.
    
    Reprojects every date onto the SAME shared grid (built once from the
    AOI, not from any individual granule) before averaging, so a date with
    partial tile coverage can't shrink the result.

    Parameters
    ----------
    entries: 
        list of {'lst_file', 'water_file', 'qc_file'} for a given tile,
        across multiple acquisition dates.
    aoi: 
        tuple of (west, south, east, north) defining the area of interest
    target_crs: 
        target coordinate reference system for reprojection
    transform: 
        affine transform for the target grid
    shape: 
        tuple of (height, width) for the target grid

    Returns
    -------
    xr.DataArray or None
        Composite LST for the tile, or None if no good pixels 
        remain after masking.
    """
    matched_arrays = []

    for entry in entries:
        lst_masked = qc_mask_lst(entry['lst_file'], entry['qc_file'])
        if lst_masked is None:
            continue

        lst_clipped = clip_and_mask_lst(lst_masked, 
                                    entry['water_file'], 
                                    entry['cloud_file'], 
                                    aoi, 
                                    target_crs=target_crs
                                    )
        if lst_clipped is None:
            continue

        lst_matched = lst_clipped.rio.reproject(target_crs, shape=shape, transform=transform)
        matched_arrays.append(lst_matched)

    if not matched_arrays:
        print(" >>> No usable granules for this tile")
        return None

    stacked = xr.concat(matched_arrays, dim='time')
    composite = stacked.mean(dim='time', skipna=True)
    composite = composite.rio.write_crs(target_crs).rio.write_nodata(np.nan)

    return composite

########################

def edge_distance_weight(da, feather_px=15):
    """
    Weight map that ramps from 0 at the edge of valid data up to 1
    after `feather_px` pixels inward. Used to feather tile boundaries.
    """
    valid = da.notnull().values
    dist = distance_transform_edt(valid)          # pixels to nearest invalid/edge pixel
    weight = np.clip(dist / feather_px, 0, 1)      # ramp 0 → 1 over feather_px pixels
    return xr.DataArray(weight, coords=da.coords, dims=da.dims)


##############################

def merge_tiles(tile_composites, target_crs, feather_px=15):
    arrays = list(tile_composites.values())
    weights = [edge_distance_weight(da, feather_px) for da in arrays]

    stacked_data = xr.concat(arrays, dim='tile')
    stacked_weights = xr.concat(weights, dim='tile').where(stacked_data.notnull(), 0)

    weighted_sum = (stacked_data.fillna(0) * stacked_weights).sum(dim='tile')
    weight_total = stacked_weights.sum(dim='tile')

    mosaic = (weighted_sum / weight_total).where(weight_total > 0)
    mosaic = mosaic.rio.write_crs(target_crs).rio.write_nodata(np.nan)

    return mosaic