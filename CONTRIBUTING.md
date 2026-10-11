# Contributing to Weather Forecasting Using LSTM

Thank you for your interest in contributing to **Weather Forecasting Using LSTM**! We welcome bug reports, feature enhancements, documentation improvements, and architectural suggestions.

This document outlines the development workflow, coding standards, and testing procedures to ensure smooth collaboration.

---

## 🛠️ Development Setup

### 1. Prerequisites
- **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Git:** Version 2.30 or newer.

### 2. Fork and Clone
Clone your fork of the repository locally:
```bash
git clone https://github.com/Sanjjith27/Weather-Forecasting.git
cd Weather-Forecasting
```

### 3. Create a Virtual Environment
Isolate your development dependencies inside a local virtual environment:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

---

## 🧪 Testing Guidelines

We prioritize high test coverage, deterministic execution, and offline test reliability.

### Running Tests Locally
Run the test suite with verbose output:
```bash
pytest -v
```

Run tests with test file filtering:
```bash
pytest tests/test_models_lstm.py -v
pytest tests/test_phase5_integration.py -v
```

### Testing Rules
1. **Zero External Network Dependencies in Unit Tests:** All tests involving external HTTP calls (such as the Open-Meteo Geocoding and Archive APIs) must mock HTTP responses using `unittest.mock.patch` or fixture mocking. Tests must pass reliably in offline environments and CI runners.
2. **Deterministic Modeling Tests:** Neural network and random operations must set explicit seeds (e.g., `seed=42`) to guarantee repeatable assertions across runs.
3. **No Metric or Data Fabrication:** Never assert on synthetic or mocked data as if it were real holdout observations. Always test realistic inputs and edge cases (e.g., missing data, invalid coordinates, single-element series).

---

## 📐 Architecture & Code Organization

Keep code modular and maintain a strict separation of concerns:

- [`src/data/`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/src/data): Open-Meteo API clients, geocoding resolvers, disk caching, schema validation, and missing observation handling.
- [`src/models/`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/src/models): Chronological dataset windowing, feature scaling, LSTM neural network architecture, persistence baselines, artifact serialization, and recursive 7-day forecasting.
- [`src/ui/`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/src/ui): Streamlit layout components, Plotly chart builders, light theme CSS styling, and user-facing metric widgets.
- [`app.py`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/app.py): Application entry point coordinating data retrieval, cached model training, and view presentation.
- [`tests/`](file:///C:/Users/Sanjjith%20B/Downloads/Weather-Forecasting/Weather-Forecasting/tests): Unit and integration test suites covering all modules.

---

## 📋 Coding Standards

1. **Code Style:** Follow PEP 8 guidelines. Use 4 spaces for indentation in Python files and UTF-8 encoding.
2. **Type Annotations & Docstrings:** Include type hints and concise docstrings on public functions, classes, and pipeline stages.
3. **Exception Handling:** Raise specific exceptions (derived from `WeatherDataError` or `WeatherModelError`) rather than bare `Exception`.
4. **No Secrets or Binary Artifacts in Git:** Never commit API credentials, personal tokens, virtual environments (`.venv`), temporary cache directories (`.pytest_cache`, `__pycache__`), or trained model weight files (`*.keras`, `*.joblib`, `*.h5`). Ensure `.gitignore` rules are respected.

---

## 🔀 Branching & Commit Workflow

1. **Branch Naming:** Create a focused feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```
2. **Commit Messages:** Follow standard Conventional Commits:
   - `feat: add relative humidity feature to LSTM input`
   - `fix: handle edge case when historical series has exactly lookback days`
   - `docs: update setup instructions for Windows PowerShell`
   - `test: add unit test for geocoding network timeout`
3. **Verification Before Committing:**
   ```bash
   pytest -v
   git diff --check
   ```
4. **Submit a Pull Request:** Push your branch and open a Pull Request against `main` with a clear description of changes and test results.
