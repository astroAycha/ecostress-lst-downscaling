import os
from pathlib import Path
from datetime import timezone, timedelta, datetime
from zoneinfo import ZoneInfo
import rasterio
import rioxarray
import numpy as np
import earthaccess


class GranuleFilter:
    """
    Class to filter ECOSTRESS granules based on acquisition time and 
    fraction of valid pixels.
    """

    def __init__(self, 
                 aoi_name: str,
                 timezone: str, 
                 valid_pixel_threshold: float = 0.7):
        
        self.aoi_name = aoi_name
        self.timezone = timezone
        self.valid_pixel_threshold = valid_pixel_threshold


    def is_day(self, 
                granule) -> bool:
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
        local_time = utc_time.astimezone(ZoneInfo(self.timezone))

        return 12 <= local_time.hour <= 18

#############
    def is_night(self, 
                   granule) -> bool:
        """
        Check if the granule's acquisition time is between 18:00 and 24:00 local time.

        Parameters
        ----------
        granule : EarthAccess granule object with UMM metadata containing TemporalExtent.
        """
        # read the time string from the granule metadata
        time_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        # parse it as UTC, and convert to local time
        utc_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        local_time = utc_time.astimezone(ZoneInfo(self.timezone))

        return 18 <= local_time.hour < 24
    
#############

    def get_valid_pixel_fraction(self,
                                 url: str, 
                                 fs) -> float:
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

    def keep_valid_granules(self, 
                            granules: list,
                            day_or_night: str = "day") -> list[dict]:
        """
        Take a list of ECOSTRESS granulues, filter out those that are outside 
        the desired time window or too cloudy, and return a list of dicts 
        containing the granule and local paths to the downloaded LST, 
        water mask, and QC mask files.

        Parameters
        ----------
        granules : list
            List of EarthAccess granule objects
        day_or_night : str, optional
            Specify whether to filter for "day" or "night" granules (default: "day")

        Returns
        -------
        list of dict
            Each dict contains:
            - "granule": the original granule object
            - "lst_file": local path to the downloaded LST GeoTIFF
            - "water_file": local path to the downloaded water mask GeoTIFF
            - "qc_file": local path to the downloaded QC mask GeoTIFF

        """
        download_dir = f"{self.aoi_name}_ecostress_data"

        # Set up an authenticated fsspec filesystem for 
        # streaming access to the LST files
        # only needed for streaming check of valid pixel fraction
        fs = earthaccess.get_fsspec_https_session()  

        good_granules = []

        for granule in granules:
            # first filter by acquisition time (local afternoon)
            if day_or_night == "day" and not self.is_day(granule):
                time_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
                print(f"Skipping (outside time window): {time_str}")
                continue
            elif day_or_night == "night" and not self.is_night(granule):
                time_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
                print(f"Skipping (outside time window): {time_str}")
                continue

            if not self.is_night(granule):
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
                valid_frac = self.get_valid_pixel_fraction(lst_url, fs)

            print(f"{granule_id[-40:]}  valid: {valid_frac:.1%}")

            if valid_frac <= self.valid_pixel_threshold:
                print(" >>> Skipping file: too many Null pixels")
                continue
            
            # Download the water, QC, and cloud mask files as well
            # Needed later for masking
            water_url = next(l for l in granule.data_links() if l.endswith("_water.tif"))
            qc_url    = next(l for l in granule.data_links() if l.endswith("_QC.tif"))
            cloud_url = next((l for l in granule.data_links() if l.endswith("_cloud.tif")), None)

            # Download the files locally using earthaccess
            #  which handles authentication and retries
            os.makedirs(download_dir, exist_ok=True)
            local_paths = earthaccess.download([lst_url, 
                                                water_url, 
                                                qc_url,
                                                cloud_url],
                                                local_path=f"./{download_dir}",
            )

            lst_dest, water_dest, qc_dest, cloud_dest = local_paths

            good_granules.append({
                "granule":    granule,
                "lst_file":   str(lst_dest),
                "water_file": str(water_dest),
                "qc_file":    str(qc_dest),
                "cloud_file": str(cloud_dest),
            })

        return good_granules