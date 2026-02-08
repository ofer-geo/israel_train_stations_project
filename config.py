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

train_station_names_eng = {"Sderot":17106,
                            "Tel Aviv Ha'Hagana":17104,
                            "Rishopn Le'Zion Moshe Dayan":17102,
                            "Hod Ha'Sharon":17100,
                             "Rishonim": 17098,
                             "Yavne West": 17096,
                             "Rosh Ha'Ayin North": 17094,
                             "Kfar Saba": 17092,
                             "Ben Gurion Airport": 17090,
                             "Lehavim Rahat": 17088,
                             'Dimona': 17086,
                             "Be'er Sheva Center": 17084,
                             "Be'er Sheva North": 17082,
                             "Kiryat Gat": 17080,
                             "Jerusalem Malha": 17078,
                             "Jerusalem Zoo": 17076,
                             "Beit Shemesh": 17074,
                             "Ashkelon": 17072,
                             "Ashdod Ad Halom": 17070,
                             "Yavne East": 17068,
                             "Be'er Ya'akov": 17066,
                             "Rehovot": 17064,
                             "Lod Ganei Aviv": 17062,
                             "Ramle": 17060,
                             "Lod": 17058,
                             "Kfar Habad": 17056,
                             "Bat Yam Komemiyut": 17054,
                             "Bat Yam - Eli Cohen Yoseftal": 17052,
                             "Holon Wolfson": 17050,
                             "Holon Junction": 17048,
                             "Tel Aviv - Ha'Shalom": 17046,
                             "Segula": 17044,
                             "Kiryat Ariye": 17042,
                             "Bne Barak": 17040,
                             "Tel Aviv Center": 17038,
                             "Tel Aviv University": 17036,
                             "Herzliya": 17034,
                             "Beit Yehoshu'a": 17032,
                             "Netanya": 17030,
                             "Hadera West": 17028,
                             "Caesarea Pardes Hana": 17026,
                             "Binyamina": 17024,
                             "Atlit": 17022,
                             "Hof Ha'Carmel": 17020,
                             "Haifa BAt Galim": 17018,
                             "Haifa Center": 17016,
                             "Nehariya": 17014,
                             "Akka": 17012,
                             "Haifa Hotzot Ha'Mifratz": 17010,
                             "Haifa Merkazit Ha'Mifratz - Kav Ha'Hof": 17008,
                             "Kiryat Haim": 17004,
                             "Modi'in Center": 17002,
                             "Pe'ate Modi'in": 17000,
                             "Netivot": 17108,
                             "Ofakim": 17109,
                             "Kfar Baruh": 17113,
                             "Afula": 17112,
                             "Ben She'an": 17111,
                             "Kfar Yehoshu'a": 17110,
                             "Netanya Kirayat Sapir": 17114,
                             "Ahihud": 17116,
                             "Carmiel": 17115,
                             "Kiryat Motzkin": 17117,
                             "Jerusalem Yitzhak Navon": 17118,
                             "Ra'nana West": 17122,
                             "Ra'nana East": 17121,
                             "Mazkeret Batya": 17120,
                             "Kiryat Malahi": 17119,
                             "Haifa - Merkazit Ha'Mifratz": 17123}


