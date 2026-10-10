# Weather Forecasting Using LSTM

An interactive, deep learning-powered weather forecasting application built with TensorFlow/Keras and Streamlit. The application uses Long Short-Term Memory (LSTM) recurrent neural networks to capture temporal patterns in historical meteorological data and generate multi-step sequential forecasts.

---

## 🎯 Project Objective

Traditional statistical methods often struggle with nonlinear dependencies and long-term sequential relationships in meteorological observations. The goal of this project is to build an end-to-end, reproducible weather forecasting system that:
1. Preprocesses time-series meteorological parameters (e.g., temperature, humidity, wind speed, pressure).
2. Trains an LSTM deep learning model capable of capturing sequential temporal dynamics.
3. Delivers actionable forecasts and intuitive visualizations through a clean, interactive Streamlit web interface.

---

## 🚀 Key Features

- **Interactive Streamlit Dashboard:** Web-based UI with city search, date filters, and responsive charts.
- **Data Preprocessing & Windowing:** Automated missing-value handling, feature scaling, and rolling sliding-window sequence generation for time-series models.
- **LSTM Deep Learning Architecture:** TensorFlow/Keras LSTM neural network designed for sequential time-series regression.
- **Forecast Evaluation & Metrics:** Comprehensive performance tracking including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ scores compared against persistence baselines.
- **Modular Design:** Clear separation of concerns between user interface, data pipelines, and deep learning modeling.

---

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **UI Framework** | [Streamlit](https://streamlit.io/) |
| **Data Visualization** | [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/) |
| **Deep Learning** | [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) |
| **Machine Learning & Preprocessing** | [Scikit-learn](https://scikit-learn.org/) |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
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

---

## 🧠 LSTM Temperature Forecasting Engine

The forecasting engine ([`src.models`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/src/models)) implements a TensorFlow/Keras LSTM model with chronological validation, leak-free scaling, persistence baseline benchmarking, and recursive multi-step forecasting:

### Methodology
1. **Chronological Train/Holdout Partitioning:** Observations are split strictly by chronological index (never randomly shuffled) to preserve natural temporal causality and prevent lookahead bias.
2. **Leakage-Free Feature Scaling:** `MinMaxScaler` is fitted strictly on training observations; the test partition and future inputs are transformed using the training parameters.
3. **Sliding-Window Sequence Generation:** Input tensors $X \in \mathbb{R}^{N \times L \times 1}$ are generated using a configurable lookback window $L$ (default 14 days) to predict target $y \in \mathbb{R}^{N \times 1}$. Context prefixing ensures zero lost observations in the holdout test set.
4. **Compact Model Architecture:**
   - `Input(shape=(lookback, 1))`
   - `LSTM(units=32, activation='tanh', return_sequences=False)`
   - `Dropout(rate=0.1)`
   - `Dense(units=16, activation='relu')`
   - `Dense(units=1)`
5. **Training & Early Stopping:** Compiled with Adam optimizer and Mean Squared Error (MSE) loss. Trains with `EarlyStopping(patience=5, restore_best_weights=True)` and fixed random seeds (42) for reproducible convergence.
6. **Honest Baseline Benchmarking:** Includes a Persistence baseline ($\hat{y}_t = y_{t-1}$) evaluated side-by-side with genuine Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the exact same holdout observations.
7. **7-Day Recursive Forecasting:** Autoregressively rolls the input window forward to produce 7 future daily temperature estimates in degrees Celsius with physical plausibility validation.
8. **Deterministic Artifact Caching:** Computes a SHA-256 fingerprint from the location, observation series signature, and model hyperparameters to persist and reload artifacts (`models/<signature>/`), avoiding redundant re-training.

### Methodological Limitations
- **Compounding Autoregressive Error:** In recursive multi-step forecasting, errors in step $t+1$ feed into inputs for step $t+2$, naturally compounding uncertainty as the horizon increases beyond 3–5 days.
- **Univariate Formulation:** The model forecasts purely from historical temperature autoregression without incorporating exogenous atmospheric parameters (e.g., pressure differentials, wind velocity, solar irradiance, cold fronts).
- **Seasonal Domain Representation:** Training on short datasets (< 1–2 years) limits the model's ability to learn full inter-annual seasonal transitions and extreme weather anomalies.

---

## 🖥️ Weather Forecasting Dashboard

The interactive web dashboard ([`app.py`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/app.py), [`src.ui`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/src/ui)) features a responsive light theme with clean typography and real-time visualization:

- **Location Search & Presets:** Quick-select major global cities (London, Tokyo, New York, Paris, Berlin, Chennai, Sydney) or search any worldwide locality with automated geocoding.
- **Historical Summary KPI Cards:** Real-time metrics for mean temperature, observed range, dataset length, and data quality.
- **Interactive Trajectory & Forecast Chart:** A Plotly chart distinguishing historical training observations, holdout ground truth, LSTM holdout estimates, and the 7-day future forecast with a shaded forecast horizon.
- **7-Day Forecast Cards:** Dedicated forecast cards for Day 1 through Day 7 displaying day-of-week, calendar date, and forecasted temperature in °C.
- **Evaluation & Benchmarking Cards:** Transparent side-by-side comparison of LSTM MAE and RMSE against the Persistence baseline with unit labeling (°C).
- **Detailed Historical Timeline:** Collapsible high-resolution view of the complete historical observation series with interpolated points flagged.
- **Educational Explanations:** Explains sliding-window lookback, recurrent memory gating, and recursive rollouts.
- **Academic Disclaimer:** Discloses experimental nature of univariate neural forecasts.

---

## 📁 Project Structure

```text
Weather-Forecasting/
├── app.py                      # Streamlit application entry point (root-level)
├── src/
│   ├── __init__.py
│   ├── data/                   # Weather data pipeline module
│   │   ├── __init__.py         # Public data interface exports
│   │   ├── client.py           # Open-Meteo historical archive client & disk cache
│   │   ├── exceptions.py       # Custom pipeline exception hierarchy
│   │   ├── geocoding.py        # City geocoding and location resolution
│   │   ├── pipeline.py         # High-level pipeline facade & WeatherDataset
│   │   ├── preprocessing.py    # Missing value handling and DataFrame construction
│   │   └── validation.py       # API schema, units, date sequence & quality checks
│   ├── models/                 # LSTM forecasting engine module
│   │   ├── __init__.py         # Public modeling interface exports
│   │   ├── baseline.py         # Persistence forecaster & genuine metrics (MAE, RMSE)
│   │   ├── dataset.py          # Chronological split, MinMaxScaler, sliding windows
│   │   ├── exceptions.py       # Model & forecasting custom exceptions
│   │   ├── lstm_model.py       # WeatherLSTMModel class, training & 7-day recursion
│   │   ├── persistence.py      # Artifact serialization & signature caching
│   │   └── pipeline.py         # train_and_forecast_pipeline coordinator
│   └── ui/                     # UI presentation layer
│       ├── __init__.py         # Public UI exports
│       ├── charts.py           # Plotly interactive chart builders
│       ├── components.py       # Reusable cards, metrics, and explanatory widgets
│       └── styles.py           # Custom CSS for light theme
├── tests/                      # Comprehensive unit test suite
│   ├── __init__.py
│   ├── test_client.py          # Client and disk caching tests
│   ├── test_geocoding.py       # Geocoding search and resolution tests
│   ├── test_models_baseline.py # Persistence baseline and metrics calculation tests
│   ├── test_models_dataset.py  # Chronological split and window dimension tests
│   ├── test_models_lstm.py     # LSTM architecture, training, and 7-day tests
│   ├── test_models_persistence.py # Artifact serialization and signature cache tests
│   ├── test_models_pipeline.py # End-to-end forecasting pipeline integration tests
│   ├── test_pipeline.py        # End-to-end data pipeline integration tests
│   ├── test_preprocessing.py   # Missing observation policy tests
│   ├── test_ui_components.py   # Plotly chart and UI styling tests
│   └── test_validation.py      # Schema, dates, and quality validation tests
├── data/                       # Local datasets (ignored by git)
├── models/                     # Trained model checkpoints & weights (ignored by git)
├── requirements.txt            # Project dependencies
├── .editorconfig               # Universal editor style formatting rules
├── .gitattributes              # Line ending normalization and file attributes
├── .gitignore                  # Repository exclusion rules
└── README.md                   # Project documentation and guide
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

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.
