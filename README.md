Train Station Activations Dashboard (Israel)

An interactive Streamlit dashboard for exploring passenger activation patterns at Israel Railways train stations.

The dashboard visualizes average daily departures, temporal patterns, and spatial context based on official public data.

What does the dashboard show?

After selecting a train station, the dashboard presents:

Monthly average daily departures
(working days only, aggregated per month)

Geographic location of the selected station

Distribution of departures by time of day
Grouped into 7 service periods (early morning, morning peak, evening peak, etc.)

Weekday distribution of departures
Share of passenger activations across Sunday–Thursday

All calculations are based on working days only, excluding:

Weekends (Fridays and Saturdays)

Israeli public holidays and their eves

Data sources

Passenger activation data
From the Israeli government open data portal:
https://data.gov.il

GTFS station data
Used for station names and geographic coordinates

The analysis in this project focuses on the year 2025.

Methodology (high level)

Daily activation columns (day_1, day_2, …) are aggregated per month

Non-service days are excluded using:

Calendar weekends

Jewish holidays (via pyluach)

Time-of-day patterns are aggregated across all working days

Percent distributions are calculated relative to total yearly activations

Running the app locally
1. Clone the repository
git clone <your-repo-url>
cd train_stations_project

2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit
streamlit run main.py


The app will open automatically in your browser at:
http://localhost:8501

Deployment

The app is designed to run on Streamlit Community Cloud directly from this repository.

Dependencies are defined in requirements.txt, and all file paths are project-relative.

Project structure
train_stations_project/
│
├── main.py              # Streamlit application
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── files/
│   └── stations.txt     # GTFS stops data
└── ...

Purpose

This project is intended for:

Exploratory data analysis

Visual comparison between train stations

Demonstrating data engineering, visualization, and dashboard design skills

Notes

The dashboard focuses on passenger activation patterns, not train schedules

All figures represent aggregated statistics, not individual trips

The project is for analytical and demonstration purposes