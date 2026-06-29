import os
from datetime import datetime
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

#############

    def get_local_hour(self, granule) -> int:
        """
        Get the granule's acquisition hour in local time.

        Parameters
        ----------
        granule : EarthAccess granule object with UMM metadata containing 
        TemporalExtent.
        """
        # read the time string from the granule metadata
        time_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        # parse it as UTC, and convert to local time
        utc_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        local_time = utc_time.astimezone(ZoneInfo(self.timezone))

        return local_time.hour
    
#############

    def get_valid_pixel_fraction(self,
                                 url: str, 
                                 fs) -> float:
        """
        Helper function to stream a GeoTIFF via HTTPS and return the 
        fraction of valid (non-masked) pixels.
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
                            local_hour_range: tuple[int, int]=(12, 18)
                            ) -> list[dict]:
        """
        Take a list of ECOSTRESS granules, filter out those outside the desired
        local acquisition time window or with too few valid pixels, and return
        a list of dicts containing local paths to the downloaded files.

        Parameters
        ----------
        granules : list
            List of EarthAccess granule objects.
        local_hour_range : tuple[int, int]
            (start_hour, end_hour) in local time, inclusive, to keep.
            Default (12, 18) captures daytime LST acquisitions.
            Use (0, 6) for nighttime, or (0, 23) to keep all.

        Returns
        -------
        list of dict
            Each dict contains:
            - "granule"   : the original granule object
            - "lst_file"  : local path to the downloaded LST GeoTIFF
            - "water_file": local path to the downloaded water mask GeoTIFF
            - "qc_file"   : local path to the downloaded QC GeoTIFF
            - "cloud_file": local path to the downloaded cloud mask GeoTIFF
        """

        download_dir = f"{self.aoi_name}_ecostress_data"
        fs = earthaccess.get_fsspec_https_session()

        start_hour, end_hour = local_hour_range
        good_granules = []

        for granule in granules:
            time_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]

            # convert UTC acquisition time to local hour
            local_hour = self.get_local_hour(granule)
            if not (start_hour <= local_hour <= end_hour):
                print(f"Skipping (hour {local_hour} outside {start_hour}–{end_hour}): {time_str}")
                continue

            lst_url    = next(l for l in granule.data_links() if l.endswith("_LST.tif"))
            granule_id = lst_url.split("/")[-1]

            # retry once with a fresh session on 403/FileNotFoundError
            for attempt in range(2):
                try:
                    with rasterio.Env(
                        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                        GDAL_HTTP_MAX_RETRY=10,
                        GDAL_HTTP_RETRY_DELAY=0.5,
                    ):
                        valid_frac = self.get_valid_pixel_fraction(lst_url, fs)
                    break  # success — exit retry loop

                except (FileNotFoundError, Exception) as e:
                    if attempt == 0:
                        print(f"  Session likely expired, refreshing and retrying...")
                        earthaccess.login(strategy="netrc")
                        fs = earthaccess.get_fsspec_https_session()
                    else:
                        print(f"  >>> Skipping {granule_id[-40:]}: streaming failed after retry ({e})")
                        valid_frac = None
                        break
            
            if valid_frac is None or valid_frac < self.valid_pixel_threshold:
                print(f"  Skipping {granule_id[-40:]}: valid fraction {valid_frac:.2f} below threshold {self.valid_pixel_threshold}")
                continue

            water_url = next(l for l in granule.data_links() if l.endswith("_water.tif"))
            qc_url    = next(l for l in granule.data_links() if l.endswith("_QC.tif"))
            cloud_url = next((l for l in granule.data_links() if l.endswith("_cloud.tif")), None)

            os.makedirs(download_dir, exist_ok=True)
            local_paths = earthaccess.download(
                [lst_url, water_url, qc_url, cloud_url],
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