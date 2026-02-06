import json
import requests
import pandas as pd
from config import BASE, RESOURCE_IDS, DEFAULT_TIMEOUT

def get_station_activations_info(stop_code: int, BASE:str, RESOURCE_ID:str,DEFAULT_TIMEOUT:int) -> pd.DataFrame | None:
    """
    Fetch station activations from data.gov.il (CKAN Datastore API) using StationId filter.
    Returns a DataFrame on success, otherwise None.
    """
    params = {
        "resource_id": RESOURCE_ID,
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


def merge_dfs_different_years(stop_code:int,RESOURCE_IDS:dict,BASE,DEFAULT_TIMEOUT) -> pd.DataFrame | None:
    dfs = list()
    for year in RESOURCE_IDS.keys():
        df_year = get_station_activations_info(stop_code,BASE,RESOURCE_IDS[year],DEFAULT_TIMEOUT)
        dfs.append(df_year)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all