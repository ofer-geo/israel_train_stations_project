import json
import requests
import pandas as pd
from config import BASE, RESOURCE_ID_ACTIVATIONS, DEFAULT_TIMEOUT

def get_station_activations_info(stop_code: int) -> pd.DataFrame | None:
    """
    Fetch station activations from data.gov.il (CKAN Datastore API) using StationId filter.
    Returns a DataFrame on success, otherwise None.
    """
    params = {
        "resource_id": RESOURCE_ID_ACTIVATIONS,
        "filters": json.dumps({"StationId": stop_code}),
    }

    try:
        r = requests.get(BASE, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()

        data = r.json()

        # CKAN-level failure
        if not data.get("success", False):
            print("Couldn't fetch station data - check spelling or try another name")
            return None

        records = data["result"]["records"]

        # Valid request but no matching rows
        if not records:
            print("Couldn't fetch station data - check spelling or try another name")
            return None

        return pd.DataFrame(records)

    except requests.exceptions.RequestException:
        print("Couldn't fetch station data - check spelling or try another name")
        return None