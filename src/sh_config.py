"""Builds a SHConfig pointed at Copernicus Data Space Ecosystem (not the old Sentinel Hub endpoints)."""
import os
from dotenv import load_dotenv
from sentinelhub import DataCollection, SHConfig

load_dotenv()

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"

# DataCollection's built-in SENTINEL2_L2A/SENTINEL1_IW definitions have service_url baked in
# to the classic Sentinel Hub deployment, which config.sh_base_url does NOT override -- so
# statistics requests silently go to services.sentinel-hub.com and 401 against CDSE credentials.
# Redefining the collections with an explicit CDSE service_url is the documented workaround.
CDSE_SENTINEL2_L2A = DataCollection.SENTINEL2_L2A.define_from(
    "CDSE_SENTINEL2_L2A", service_url=CDSE_BASE_URL
)
CDSE_SENTINEL1_IW = DataCollection.SENTINEL1_IW.define_from(
    "CDSE_SENTINEL1_IW", service_url=CDSE_BASE_URL
)


def get_config() -> SHConfig:
    config = SHConfig()
    config.sh_client_id = os.environ["SH_CLIENT_ID"]
    config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    return config
