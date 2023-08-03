"""
03.08.2023
Data Dashboard Wisconsin Breast Cancer Data
Bayer Challenge Team D Life
"""

import pandas as pd
import streamlit as st


def read_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


df = read_data("data.csv")
df = df.dropna(axis=1, how='all')


def create_tabs():
    tab1, tab2, tab3 = st.tabs(['Overview', 'AI', 'Predictions'])

    with tab1:
        st.header("Data Overview")

    with tab2:
        st.header("Ask the AI")

    with tab3:
        st.header("Predictions")


def main():
    create_tabs()


if __name__ == "__main__":
    main()
