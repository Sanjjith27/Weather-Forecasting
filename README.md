# Weather Forecasting Using LSTM

An interactive, deep learning-powered weather forecasting application built with PyTorch and Streamlit. The application uses Long Short-Term Memory (LSTM) recurrent neural networks to capture temporal patterns in historical meteorological data and generate multi-step sequential forecasts.

---

## 🎯 Project Objective

Traditional statistical methods often struggle with nonlinear dependencies and long-term sequential relationships in meteorological observations. The goal of this project is to build an end-to-end, reproducible weather forecasting system that:
1. Preprocesses time-series meteorological parameters (e.g., temperature, humidity, wind speed, pressure).
2. Trains an LSTM deep learning model capable of capturing sequential temporal dynamics.
3. Delivers actionable forecasts and intuitive visualizations through a clean, interactive Streamlit web interface.

---

## 🚀 Planned Features

- **Interactive Streamlit Dashboard:** Web-based UI with parameter selection, date filters, and responsive charts.
- **Data Preprocessing & Windowing:** Automated missing-value handling, feature scaling, and rolling sliding-window sequence generation for time-series models.
- **LSTM Deep Learning Architecture:** PyTorch-based LSTM neural network designed for sequential time-series regression.
- **Forecast Evaluation & Metrics:** Comprehensive performance tracking including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ scores with visual actual-vs-predicted comparisons.
- **Modular Design:** Clear separation of concerns between user interface, data pipelines, and deep learning modeling.

---

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **UI Framework** | [Streamlit](https://streamlit.io/) |
| **Deep Learning** | [PyTorch](https://pytorch.org/) |
| **Machine Learning & Preprocessing** | [Scikit-learn](https://scikit-learn.org/) |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Data Visualization** | [Matplotlib](https://matplotlib.org/) |
| **HTTP & APIs** | [Requests](https://requests.readthedocs.io/) |
| **Environment & Configuration** | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| **Testing** | [pytest](https://docs.pytest.org/) |

---

## 🛰️ Historical Weather Data Pipeline

The data ingestion pipeline ([`src.data`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/src/data)) handles location resolution, historical observations retrieval, and dataset quality validation:

- **City Geocoding:** Resolves city names to geographical coordinates via the Open-Meteo Geocoding API (`search_locations`, `geocode_city`).
- **Historical Observations API:** Fetches historical daily mean temperatures in Celsius (°C) using the dedicated Open-Meteo Archive endpoint (`https://archive-api.open-meteo.com/v1/archive`).
- **Data Validation & Quality Assurance:**
  - Validates API payload schema, HTTP status codes, and Celsius temperature units (`validate_api_response_schema`).
  - Enforces strictly chronological, duplicate-free daily dates (`validate_date_sequence`).
  - Enforces physical plausibility bounds ($-90^\circ\text{C}$ to $60^\circ\text{C}$) and sensor flatline / zero-variance detection.
  - Requires minimum sample size threshold (e.g., 30+ observations) before passing to LSTM.
- **Missing Observations Handling:**
  - Transparently manages missing readings without fabricating synthetic weather measurements.
  - Supports `interpolate` (linear interpolation bounded to isolated gaps $\le 2$ consecutive days with `is_interpolated` flag tracking), `drop` (removes unobserved dates), or `error` (rejects missing data).
- **Persistent Disk Caching:** Deterministic SHA-256 parameter-hashed local JSON cache reduces duplicate network requests and respects API rate limits.
- **Offline Test Suite:** 100% mocked unit tests verify geocoding, client caching, schema validation, and preprocessing without requiring internet connectivity.

### Example Pipeline Usage

```python
from src.data import fetch_weather_by_city

# Fetch, validate, and preprocess 1 year of daily mean temperature observations
dataset, location = fetch_weather_by_city(
    city_name="Berlin",
    start_date="2023-01-01",
    end_date="2023-12-31",
    missing_strategy="interpolate",
    min_samples=30,
)

print(f"Location: {location.display_name()}")
print(f"Dataset summary: {dataset.summary()}")
print(dataset.df.head())
```

---

## 📁 Project Structure

```text
Weather-Forecasting/
├── app.py                   # Streamlit application entry point (root-level)
├── src/
│   ├── __init__.py
│   └── data/                # Weather data pipeline module
│       ├── __init__.py      # Public interface exports
│       ├── client.py        # Open-Meteo historical archive client & disk cache
│       ├── exceptions.py    # Custom pipeline exception hierarchy
│       ├── geocoding.py     # City geocoding and location resolution
│       ├── pipeline.py      # High-level pipeline facade & WeatherDataset
│       ├── preprocessing.py # Missing value handling and DataFrame construction
│       └── validation.py    # API schema, units, date sequence & quality checks
├── tests/                   # Mocked unit test suite
│   ├── __init__.py
│   ├── test_client.py       # Client and disk caching tests
│   ├── test_geocoding.py    # Geocoding search and resolution tests
│   ├── test_pipeline.py     # End-to-end pipeline integration tests
│   ├── test_preprocessing.py# Missing observation policy tests
│   └── test_validation.py   # Schema, dates, and quality validation tests
├── data/                    # Local datasets (ignored by git)
├── models/                  # Trained model checkpoints & weights (ignored by git)
├── requirements.txt         # Project dependencies
├── .editorconfig            # Universal editor style formatting rules
├── .gitattributes           # Line ending normalization and file attributes
├── .gitignore               # Repository exclusion rules
└── README.md                # Project documentation and guide
```

---

## ⚡ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Sanjjith27/Weather-Forecasting.git
cd Weather-Forecasting
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Test Suite
```bash
pytest -v
```

### 5. Run the Application
```bash
streamlit run app.py
```
