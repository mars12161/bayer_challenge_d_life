# Importing libs

import pandas as pd
import time
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import dash

@st.cache_data
def read_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

# Exploratory data analysis


df = read_data("data.csv")
df = df.dropna(axis=1, how='all')
# df.head()

# Function to check data (info, size, null values etc)
def data_checks(dataframe: pd.DataFrame) -> dict():
    """
    perform checks in the dataset

    args:
        df (pd.DataFrame): the DataFrame which we are performing checks

    returns:
        dictionary of checks

    """
    checks = {
        "info": dataframe.info(),
        # "types":dataframe.dtypes(),
        "shape": dataframe.shape,
        "uniqueness": dataframe.apply(lambda x: len(x.unique())).sort_values(ascending=False).head(10),
        "missing_values": dataframe.isnull().sum(),
        "duplicates": dataframe.duplicated().sum(),
        "data_snapshot": dataframe.head()

    }
    return checks


#data_checks(df)


#description = df.describe()
#print(description)


def data_overview():
    st.subheader('Data Overview:')
    st.write(df.head())


def plot_summary():
    st.subheader('Summary Statistics:')
    st.write(df.describe())


def write_checks():
    st.subheader('Data Checks:')
    st.write(data_checks(df))


def plot_distribution():
    st.subheader('Class Distribution:')
    counts = df['diagnosis'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig, clear_figure=True)


def symmetry_boxplot():
    st.subheader('Symmetry vs. Diagnosis (Box Plot):')
    sns.boxplot(x='diagnosis', y='symmetry_mean', data=df)
    plt.title('Symmetry vs. Diagnosis')
    st.pyplot(plt, clear_figure=True)


def symmetry_violinplot():
    st.subheader('Symmetry vs. Diagnosis (Violin Plot):')
    sns.violinplot(x='diagnosis', y='symmetry_mean', data=df)
    plt.title('Symmetry vs. Diagnosis')
    st.pyplot(plt, clear_figure=True)


def correlation_heatmap():
    subset_df = df.iloc[:, 2:12]
    corr_matrix = subset_df.corr()

    st.subheader('Correlation Heatmap:')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)


# Create streamlit dashboard

def main():
    st.title('Wisconsin Breast Cancer Data Analysis')
    st.write('Exploratory Data Analysis of Wisconsin Breast Cancer Data')

    data_overview()
    plot_summary()
    # write_checks()

    plot_distribution()
    symmetry_boxplot()
    symmetry_violinplot()

    correlation_heatmap()


if __name__ == '__main__':
    main()
    # run program like this: streamlit run data_dashboard_tryout.py

