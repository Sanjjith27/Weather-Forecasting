"""Custom CSS styles for the Weather Forecasting dashboard's premium light theme."""

LIGHT_THEME_CSS = """
<style>
/* --------------------------------------------------------------------------
   Global Typography and Base Layout
--------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
}

/* Light background canvas */
.stApp {
    background-color: #f8fafc;
}

/* --------------------------------------------------------------------------
   Header and Typography
--------------------------------------------------------------------------- */
.main-title {
    font-size: 2.25rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1e293b;
    letter-spacing: -0.01em;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

/* --------------------------------------------------------------------------
   Metric and Summary Cards
--------------------------------------------------------------------------- */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* --------------------------------------------------------------------------
   7-Day Forecast Cards
--------------------------------------------------------------------------- */
.forecast-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}

.forecast-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.9rem 0.75rem;
    text-align: center;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
    transition: border-color 0.15s ease, transform 0.15s ease;
}

.forecast-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
}

.forecast-day {
    font-size: 0.8rem;
    font-weight: 600;
    color: #0284c7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.forecast-date {
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 0.45rem;
}

.forecast-temp {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.forecast-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 9999px;
    background-color: #f1f5f9;
    color: #475569;
    margin-top: 0.4rem;
}

/* --------------------------------------------------------------------------
   Academic Disclaimer Banner
--------------------------------------------------------------------------- */
.disclaimer-banner {
    background-color: #f8fafc;
    border-left: 4px solid #0284c7;
    border-radius: 4px 8px 8px 4px;
    padding: 0.85rem 1.15rem;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
}

.disclaimer-banner strong {
    color: #0f172a;
}

/* --------------------------------------------------------------------------
   Explanation and Code Cards
--------------------------------------------------------------------------- */
.explanation-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
}

.explanation-card h4 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 0.5rem;
}

.explanation-card p, .explanation-card li {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.6;
}
</style>
"""
