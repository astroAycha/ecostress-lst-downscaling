import xarray as xr
import numpy as np
import rioxarray

def get_good_quality_mask(qc_da: xr.DataArray) -> xr.DataArray:
    """
    Build a boolean mask from the ECOSTRESS QC band.
    Returns True where pixels are good quality (bits 0-1 == '00').
    
    Parameters
    ----------
    qc_da : xr.DataArray
        QC band loaded from the ECOv002*_QC.tif file 
        Values are 16-bit integers where bits 0-1 encode quality flags:
        '00' = good quality pixels
        '01', '10', '11' = various issues (clouds, missing data, etc.)
    
    Returns
    -------
    xr.DataArray
        Boolean mask, True = keep, False = discard
    """

    # read unique QC values
    unique_vals = np.unique(qc_da.values.tolist())

    # convert into binary and check last 2 bits
    # keep only the good stuff where bits 0-1 == '00'
    good_vals = [q for q in unique_vals if np.binary_repr(q.astype(int), width=16)[-2:] == "00"]

    # return the mask where QC values are in the good_vals list
    return qc_da.isin(good_vals)


def qc_mask_lst(lst_file: str, qc_file: str) -> xr.DataArray | None:
    """
    Load LST file, apply QC mask, and return a masked DataArray ready for stacking.
    Returns None if no good pixels remain after masking.

    Parameters
    ----------
    lst_file : str
        Path to the ECOv002*_LST.tif file.
    qc_file : str
        Path to the ECOv002*_QC.tif file for the same granule.

    Returns
    -------
    xr.DataArray or None
    """
    # load LST and QC as xarray DataArrays with rioxarray
    # ensuring they are masked and squeezed to 2D
    lst = rioxarray.open_rasterio(lst_file, masked=True).squeeze()
    qc  = rioxarray.open_rasterio(qc_file,  masked=True).squeeze()

    # align QC to LST grid (might not be necessary if already aligned)
    qc = qc.rio.reproject_match(lst)

    # apply QC mask to LST
    good_mask = get_good_quality_mask(qc)
    lst_masked = lst.where(good_mask)

    # report fraction of valid pixels after masking
    valid_frac = float(lst_masked.notnull().mean())
    print(f"  QC mask applied >>> {valid_frac:.1%} pixels retained")

    if valid_frac == 0:
        print(" >>> Skipping: no good pixels after QC masking")
        return None

    return lst_masked

###########################
def clip_and_water_mask_lst(
    lst_masked: xr.DataArray,
    water_file: str,
    aoi: tuple,
    target_crs: str = "EPSG:32617",
) -> xr.DataArray | None:
    """
    Clip, reproject, and apply water mask to a QC-masked LST DataArray.

    Parameters
    ----------
    lst_masked : xr.DataArray
        Output of qc_mask_lst — QC-masked LST in native CRS.
    water_file : str
        Path to the ECOv002*_water.tif file for the same granule.
    aoi : tuple
        Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    target_crs : str
        CRS to reproject into. Default is UTM 17N for Toronto.

    Returns
    -------
    xr.DataArray or None
    """
    # read the water mask and reproject to match LST CRS
    water = rioxarray.open_rasterio(water_file, masked=True).squeeze()
    water = water.rio.reproject_match(lst_masked)

    # mask water pixels (water == 1)
    lst_land = lst_masked.where(water != 1)

    # reproject to target CRS
    lst_reproj = lst_land.rio.reproject(target_crs)

    # clip to the AOI
    west, south, east, north = aoi
    lst_clipped = lst_reproj.rio.clip_box(
        minx=west,
        miny=south,
        maxx=east,
        maxy=north,
        crs="EPSG:4326", 
    )

    land_frac = float(lst_clipped.notnull().mean())

    if land_frac == 0:
        print(" >>> Skipping: no good pixels after preprocessing")
        return None

    return lst_clipped