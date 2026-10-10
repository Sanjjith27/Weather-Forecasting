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
| **Environment & Configuration** | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## 📁 Project Structure

```text
Weather-Forecasting/
├── app.py              # Streamlit application entry point (root-level)
├── src/                # Core application and modeling modules (Phase 2+)
│   ├── data/           # Dataset loading, cleaning, and preprocessing
│   ├── models/         # PyTorch LSTM architecture and training pipelines
│   └── utils/          # Evaluation metrics and visualization helpers
├── data/               # Local datasets (raw / processed - ignored by git)
├── models/             # Trained model checkpoints & weights (ignored by git)
├── requirements.txt    # Initial project dependencies
├── .editorconfig       # Universal editor style formatting rules
├── .gitattributes      # Line ending normalization and file attributes
├── .gitignore          # Repository exclusion rules
└── README.md           # Project documentation and guide
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

### 4. Run the Application
```bash
streamlit run app.py
```
