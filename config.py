from typing import Literal

# CKAN Datastore endpoint (data.gov.il)
BASE = "https://data.gov.il/api/3/action/datastore_search"

# Passenger activations dataset resource id
RESOURCE_ID_ACTIVATIONS = "b2c6b258-4638-4f8e-bcad-600f0cdfb449"

DEFAULT_TIMEOUT = 30

RESOURCE_IDS = {
    2023:"e265857e-9f53-419f-ad0b-860d2bf6fbb8",
    2024:"51703b73-c27b-497e-8701-ea979a0c3835",
    2025:"b2c6b258-4638-4f8e-bcad-600f0cdfb449"
}

YEARS = list(RESOURCE_IDS.keys())

TemporalInterval = ["day", "month", "year"]


