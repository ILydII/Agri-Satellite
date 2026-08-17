"""Builds a SHConfig pointed at Copernicus Data Space Ecosystem (not the old Sentinel Hub endpoints)."""
import os
from dotenv import load_dotenv
from sentinelhub import SHConfig

load_dotenv()


def get_config() -> SHConfig:
    config = SHConfig()
    config.sh_client_id = os.environ["SH_CLIENT_ID"]
    config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    return config
