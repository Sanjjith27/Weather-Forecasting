"""Custom CSS design system for the Weather Forecasting dashboard's premium light theme."""

LIGHT_THEME_CSS = """
<style>
/* --------------------------------------------------------------------------
   Global Typography and Base Layout
--------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Light background canvas */
.stApp {
    background-color: #f8fafc;
}

/* Tabular figures for all numeric displays */
.metric-value, .forecast-temp, .metric-sub, .benchmark-value {
    font-variant-numeric: tabular-nums;
}

/* Remove awkward top padding in main Streamlit block */
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* --------------------------------------------------------------------------
   Header and Product Identity
--------------------------------------------------------------------------- */
.header-container {
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e2e8f0;
}

.brand-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #0284c7;
    background-color: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 9999px;
    padding: 0.2rem 0.65rem;
    margin-bottom: 0.5rem;
}

.brand-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #0284c7;
}

.main-title {
    font-size: 2.15rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.4rem 0;
}

.sub-title {
    font-size: 0.9rem;
    color: #64748b;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.75rem;
}

.sub-title-item {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.sub-title strong {
    color: #1e293b;
    font-weight: 600;
}

.sub-title-divider {
    color: #cbd5e1;
}

/* --------------------------------------------------------------------------
   Section Headers
--------------------------------------------------------------------------- */
.section-header-wrap {
    margin-top: 2rem;
    margin-bottom: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.015em;
    margin: 0;
}

.section-caption {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0;
}

/* --------------------------------------------------------------------------
   Metric and Summary Cards
--------------------------------------------------------------------------- */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px -2px rgba(15, 23, 42, 0.05), 0 2px 4px -1px rgba(15, 23, 42, 0.03);
    border-color: #cbd5e1;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.025em;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.35rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
}

.status-pill-success {
    background-color: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

.status-pill-info {
    background-color: #f0f9ff;
    color: #0284c7;
    border: 1px solid #bae6fd;
}

/* --------------------------------------------------------------------------
   7-Day Forecast Cards
--------------------------------------------------------------------------- */
.forecast-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
    margin-bottom: 1.75rem;
}

.forecast-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 0.75rem;
    text-align: center;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
    transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
}

.forecast-card:hover {
    border-color: #0284c7;
    transform: translateY(-2px);
    box-shadow: 0 6px 12px -2px rgba(15, 23, 42, 0.06);
}

.forecast-day {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0284c7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.forecast-date {
    font-size: 0.74rem;
    color: #64748b;
    margin-top: 0.15rem;
    margin-bottom: 0.6rem;
}

.forecast-temp {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.025em;
    line-height: 1.15;
}

.forecast-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    background-color: #f1f5f9;
    color: #475569;
    margin-top: 0.5rem;
}

/* --------------------------------------------------------------------------
   Model Benchmarking Cards
--------------------------------------------------------------------------- */
.delta-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 6px;
    margin-top: 0.25rem;
}

.delta-tag-neutral {
    background-color: #f1f5f9;
    color: #475569;
}

.delta-tag-accent {
    background-color: #e0f2fe;
    color: #0369a1;
}

/* --------------------------------------------------------------------------
   Methodology & Explanation Cards
--------------------------------------------------------------------------- */
.methodology-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
    margin-top: 0.75rem;
}

.methodology-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.15rem;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.02);
}

.methodology-card h4 {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
    margin-top: 0;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.methodology-card p {
    font-size: 0.8rem;
    color: #475569;
    line-height: 1.55;
    margin: 0;
}

.explanation-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
}

.explanation-card h4 {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 0.5rem;
}

.explanation-card p, .explanation-card li {
    font-size: 0.84rem;
    color: #475569;
    line-height: 1.6;
}

/* --------------------------------------------------------------------------
   Academic Disclaimer Banner
--------------------------------------------------------------------------- */
.disclaimer-banner {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #0284c7;
    border-radius: 8px;
    padding: 0.9rem 1.25rem;
    margin-top: 1.75rem;
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.55;
}

.disclaimer-banner strong {
    color: #0f172a;
}

/* --------------------------------------------------------------------------
   Minimalist Footer
--------------------------------------------------------------------------- */
.dashboard-footer {
    border-top: 1px solid #e2e8f0;
    padding-top: 1.5rem;
    margin-top: 2.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.78rem;
    color: #64748b;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
}

.dashboard-footer a {
    color: #0284c7;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s ease;
}

.dashboard-footer a:hover {
    color: #0369a1;
    text-decoration: underline;
}

/* --------------------------------------------------------------------------
   Streamlit Sidebar & Widget Refinements
--------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

[data-testid="stSidebar"] hr {
    margin: 1.25rem 0 !important;
    border-color: #f1f5f9 !important;
}

.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
    margin-top: 1.1rem;
    margin-bottom: 0.6rem;
}

/* Primary Button Styling */
button[kind="primary"] {
    background-color: #0284c7 !important;
    border: 1px solid #0369a1 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(2, 132, 199, 0.15) !important;
}

button[kind="primary"]:hover {
    background-color: #0369a1 !important;
    border-color: #075985 !important;
    box-shadow: 0 3px 6px rgba(2, 132, 199, 0.25) !important;
    transform: translateY(-1px);
}

button[kind="primary"]:active {
    transform: scale(0.98);
}
</style>
"""
