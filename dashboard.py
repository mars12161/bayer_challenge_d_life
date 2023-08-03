"""
03.08.2023
Data Dashboard Wisconsin Breast Cancer Data
Bayer Challenge Team D Life
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI


def read_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


df = read_data("data.csv")
df = df.dropna(axis=1, how='all')


def ask_pandas():
    llm = OpenAI(api_token='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')
    pandasai = PandasAI(llm)
    with st.form("Question"):
        question = st.text_input("Question", value="", type="default")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with st.spinner("Thinking..."):
                answer = pandasai.run(df, prompt=question)

                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
                st.write(answer)



def create_tabs():
    tab1, tab2, tab3 = st.tabs(['Overview', 'AI', 'Predictions'])

    with tab1:
        st.header("Data Overview")

    with tab2:
        st.header("Ask the AI")
        st.write("Here you can ask the AI a question about the data")
        ask_pandas()

    with tab3:
        st.header("Predictions")


def main():
    create_tabs()


if __name__ == "__main__":
    main()
