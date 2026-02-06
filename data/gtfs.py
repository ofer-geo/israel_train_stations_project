def get_trains_stations_info(df_stations: pd.DataFrame):
    """
    Filter GTFS stops to (approx.) Israel Railways station stop_code range,
    then return:
    - dict: {stop_name -> stop_code}
    - list: station names (for UI selector)
    """
    df_train_stations = df_stations[df_stations["stop_code"].between(17000, 17200, inclusive="both")]
    train_stations_dict = dict(zip(df_train_stations["stop_name"], df_train_stations["stop_code"]))
    train_station_names = sorted(list(train_stations_dict.keys()))
    return train_stations_dict, train_station_names