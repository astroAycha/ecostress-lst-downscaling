from pathlib import Path
from datetime import timezone, timedelta, datetime
from zoneinfo import ZoneInfo
import rasterio
import rioxarray
import numpy as np
import earthaccess

# dir to save downloaded files (LST, water mask, QC mask)
Path("./ecostress_data").mkdir(parents=True, exist_ok=True)

# Toronto timezone for local time filtering
# TODO: ideally this should be parameterized 
toronto_tz = ZoneInfo("America/Toronto")


def is_afternoon(granule) -> bool:
    """
    Check if the granule's acquisition time is between 12:00 and 18:00 local time.

    Parameters
    ----------
    granule : EarthAccess granule object with UMM metadata containing TemporalExtent.
    """
    # read the time string from the granule metadata
    time_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    # parse it as UTC, and convert to local time
    utc_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    local_time = utc_time.astimezone(toronto_tz)

    return 12 <= local_time.hour <= 18

#############

def get_valid_pixel_fraction(url: str, fs) -> float:
    """
    Helper function to stream a GeoTIFF via HTTPS and return the fraction of 
    valid (non-masked) pixels.
    Relies on the caller having set up a rasterio.Env() context.

    Parameters
    ----------
    url : str
        HTTPS URL to the GeoTIFF.
    fs : fsspec filesystem
        Authenticated filesystem object from earthaccess.

    Returns
    -------
    float
        Fraction of valid pixels in [0, 1].
    """
    with rioxarray.open_rasterio(fs.open(url), masked=True) as src:
        da = src.squeeze().load()
        return float(da.notnull().mean())


#######

def keep_valid_granules(granules: list) -> list[dict]:
    """
    Take a list of ECOSTRESS granulues, filter out those that are outside 
    the desired time window or too cloudy, and return a list of dicts 
    containing the granule and local paths to the downloaded LST, 
    water mask, and QC mask files.

    Parameters
    ----------
    granules : list
        List of EarthAccess granule objects.
    
    Returns
    -------
    list of dict
        Each dict contains:
        - "granule": the original granule object
        - "lst_file": local path to the downloaded LST GeoTIFF
        - "water_file": local path to the downloaded water mask GeoTIFF
        - "qc_file": local path to the downloaded QC mask GeoTIFF

    """
    # Set up an authenticated fsspec filesystem for 
    # streaming access to the LST files
    # only needed for streaming check of valid pixel fraction
    fs = earthaccess.get_fsspec_https_session()  

    good_granules = []

    for granule in granules:
        # first filter by acquisition time (local afternoon)
        if not is_afternoon(granule):
            time_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            print(f"Skipping (outside time window): {time_str}")
            continue

        lst_url = next(l for l in granule.data_links() if l.endswith("_LST.tif"))
        granule_id = lst_url.split("/")[-1]

        # Now, check the fraction of valid pixels by streaming the LST file 
        # and reading it as a masked array
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY=10,
            GDAL_HTTP_RETRY_DELAY=0.5,
        ):
            valid_frac = get_valid_pixel_fraction(lst_url, fs)

        print(f"{granule_id[-40:]}  valid: {valid_frac:.1%}")

        if valid_frac <= 0.7:
            print(" >>> Skipping file (too many masked pixels)")
            continue
        
        # Download the water and QC mask files as well
        # Needed later for masking
        water_url = next(l for l in granule.data_links() if l.endswith("_water.tif"))
        qc_url    = next(l for l in granule.data_links() if l.endswith("_QC.tif"))

        # Download the files locally using earthaccess
        #  which handles authentication and retries
        local_paths = earthaccess.download(
            [lst_url, water_url, qc_url],
            local_path="./ecostress_data",
        )

        lst_dest, water_dest, qc_dest = local_paths

        good_granules.append({
            "granule":    granule,
            "lst_file":   str(lst_dest),
            "water_file": str(water_dest),
            "qc_file":    str(qc_dest),
        })

    return good_granules