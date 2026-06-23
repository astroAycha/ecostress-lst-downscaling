import xarray as xr
import numpy as np
import rioxarray
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster


def get_good_quality_mask(qc_da: xr.DataArray) -> xr.DataArray:
    """
    Build a boolean mask from the ECOSTRESS QC band using bitwise operations.
    Returns True where pixels are good or acceptable quality (bits 0-1 == 00 or 01).

    The QC band encodes quality information in binary. We only care about the
    lowest 2 bits (bits 0 and 1), which encode the main quality flag:
        00 -> good quality          -> keep (True)
        01 -> acceptable quality    -> keep (True)
        10 -> poor quality          -> discard (False)
        11 -> bad / missing data    -> discard (False)

    Parameters
    ----------
    qc_da : xr.DataArray
        QC band loaded from the ECOv002*_QC.tif file.

    Returns
    -------
    xr.DataArray
        Boolean mask, True = keep, False = discard.
    """
    # NaN pixels (no QC data) are filled with 255 = 0b11111111
    # so they evaluate to bits 0-1 = 11 and get discarded safely
    qc_int = qc_da.fillna(255).astype(np.uint16)

    # AND with 0b11 (= 3) zeroes out all bits except bits 0 and 1
    # e.g. 0b10110100 & 0b11 = 0b00 → good
    #      0b10110101 & 0b11 = 0b01 → acceptable
    #      0b10110110 & 0b11 = 0b10 → poor → discard
    bits_0_1 = qc_int & 0b11

    # keep pixels where the lowest 2 bits are 00 or 01 (values 0 and 1)
    return bits_0_1 < 2


def qc_mask_lst(lst_file: str, 
                qc_file: str) -> xr.DataArray | None:
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

def clip_and_mask_lst(lst_masked: xr.DataArray,
                        water_file: str,
                        cloud_file: str,
                        aoi: tuple,
                        target_crs: str,
                        cloud_cover_frac: int = 10) -> xr.DataArray | None:
    """
    Clip, reproject, and apply water and cloud masks to 
    a QC-masked LST DataArray

    Parameters
    ----------
    lst_masked : xr.DataArray
        LST DataArray after QC masking using `qc_mask_lst()`
    water_file : str
        Path to the water mask raster file
    cloud_file : str
        Path to the cloud mask raster file
    aoi : tuple
        (west, south, east, north) bounding box of the area of interest
          in EPSG:4326
    target_crs : str
        Target CRS for reprojection (e.g. "EPSG:32610" for UTM 10N)
    cloud_cover_frac : int
        Maximum allowed cloud cover fraction (0-100) for the granule. 
        If the cloud mask indicates more than this fraction of pixels 
        are cloudy, the granule is skipped.
    
    Returns
    -------
    xr.DataArray or None
         Preprocessed LST DataArray clipped to AOI, 
         with water pixels masked out and reprojected to target CRS. 
         Returns None if no good pixels remain after processing.
    """

    # read the water mask and reproject to match LST CRS
    water = rioxarray.open_rasterio(water_file, masked=True).squeeze()
    water = water.rio.reproject_match(lst_masked)

    # read the cloud mask and reproject to match LST CRS
    cloud = rioxarray.open_rasterio(cloud_file, masked=True).squeeze()
    cloud = cloud.rio.reproject_match(lst_masked)

    # mask out water and cloud pixels from the LST
    # keep only land pixels that are not cloudy
    lst_land = lst_masked.where((water != 1) & (cloud != 1))

    lst_reproj = lst_land.rio.reproject(target_crs)

    # clip to AOI and check if any good pixels remain
    west, south, east, north = aoi
    try:
        lst_clipped = lst_reproj.rio.clip_box(
            minx=west, miny=south, maxx=east, maxy=north, crs="EPSG:4326")
        
    except (NoDataInBounds, OneDimensionalRaster):
        print(" >>> Skipping: granule does not meaningfully overlap AOI")
        return None
    
    # check cloud cover fraction in the AOI
    if cloud_cover_frac is not None:
        total_pixels = lst_clipped.size
        cloudy_pixels = (cloud.rio.clip_box(
            minx=west, miny=south, maxx=east, maxy=north, crs="EPSG:4326") == 1).sum().item()
        cloud_cover = (cloudy_pixels / total_pixels) * 100 if total_pixels > 0 else 100
        print(f"  Cloud cover in AOI: {cloud_cover:.1f}%")
        if cloud_cover > cloud_cover_frac:
            print(f" >>> Skipping: cloud cover {cloud_cover:.1f}% exceeds threshold of {cloud_cover_frac}%")
            return None

    land_frac = float(lst_clipped.notnull().mean())
    if land_frac == 0:
        print(" >>> Skipping: no good pixels after preprocessing")
        return None

    return lst_clipped