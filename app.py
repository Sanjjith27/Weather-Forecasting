"""Weather Forecasting Using LSTM - Streamlit Application Entry Point."""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Weather Forecasting Using LSTM",
        page_icon="⛅",
        layout="wide",
    )

    st.title("Weather Forecasting Using LSTM")
    st.markdown(
        """
        Welcome to the **Weather Forecasting Using LSTM** application.

        This system is designed to forecast meteorological time-series observations
        (such as temperature, humidity, and precipitation) using Long Short-Term
        Memory (LSTM) recurrent neural networks.
        """
    )


if __name__ == "__main__":
    main()
