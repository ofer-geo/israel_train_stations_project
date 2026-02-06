import pandas as pd
import folium

def station_location_on_map(df: pd.DataFrame, stop_code: int, station_name: str):
    """
    Folium map with a single marker at the selected station coordinates.
    """
    lat, lon = df.loc[df["stop_code"] == stop_code, ["stop_lat", "stop_lon"]].iloc[0]

    m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap")
    folium.Marker(
        location=[lat, lon],
        popup=station_name,
        tooltip=station_name,
        icon=folium.Icon(color="red", icon="train", prefix="fa"),
    ).add_to(m)

    return m