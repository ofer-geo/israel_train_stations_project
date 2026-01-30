# Train Station Activations Dashboard (Streamlit)

An interactive **Streamlit dashboard** for exploring **passenger activation patterns at Israel Railways train stations** based on official public transportation data.

Select a station to analyze monthly trends, temporal usage patterns, weekday distributions, and geographic location.

---

## Purpose
This project is intended for:
- Exploratory analysis of public transportation demand
- Visual comparison between train stations
- Demonstrating data engineering, visualization, and dashboard design skills

---

## Features

- **Station selector**
  - Choose any Israel Railways train station from the GTFS dataset

- **Monthly average daily departures**
  - Aggregated per month
  - Calculated for **working days only**

- **Time-of-day distribution**
  - Passenger departures grouped into **7 service periods**
  - Displayed as percentage share

- **Weekday distribution**
  - Share of passenger activations across **Sunday–Thursday**

- **Geographic visualization**
  - Interactive station location map (Folium)

---

## Methodology

- Daily activation columns (`day_1`, `day_2`, …) are aggregated per month
- Non-service days are excluded:
  - Fridays and Saturdays
  - Israeli public holidays and their eves
- Time-of-day analysis aggregates all working days across the year
- Percent distributions are calculated relative to total yearly activations

The analysis focuses on **the year 2025**.

---

## Tech Stack

- **Python**
- **Streamlit** (dashboard framework)
- **Pandas** (data processing)
- **NumPy** (numerical operations)
- **Requests** (API access)
- **Plotly Express** (interactive charts)
- **Matplotlib / Seaborn** (static charts)
- **Folium** (interactive maps)
- **streamlit-folium** (Folium integration)
- **pyluach** (Jewish holiday calendar)

---

## Data Sources

- **Passenger activation data**
  - Israeli Government Open Data Portal  
    https://data.gov.il

- **GTFS station data**
  - Used for station names and geographic coordinates

---

## Project Structure

```text
train_stations_project/
│
├── main.py              # Streamlit application
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── files/
│   └── stations.txt     # GTFS stops data
└── ...
```

## Getting Started

### 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd train_stations_project
```

### 2) Install dependencies
```bash
Install dependencies
```
### 3) Run the application
```bash
streamlit run main.py
```

The app will open automatically in your browser at:
```arduino
http://localhost:8501
```

---

## Deployment
This project is designed to run on *Streamlit Community Cloud* directly from this repository.

- Dependencies are defined in requirements.txt

- All file paths are project-relative

- No local configuration is required 
After deployment, the app can be shared via a public URL.
