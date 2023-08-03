import streamlit as st
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# readings and prep data
df = pd.read_csv("data.csv")
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
df = df.drop(columns=["id", "Unnamed: 32"])
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
corr_diagnosis = df.corr()['diagnosis'].sort_values(ascending=False)
mydict = corr_diagnosis.to_dict()

###
st.title(":bar_chart: Minimal dashboard team d-life")
st.header("Data at a glance")

col1, col2 = st.columns((2))

with col1:
    result = st.selectbox("**Select your feature:**",list(mydict.keys()))

with col2:
    st.write(f"#### {result} correlation with diagnosis: ")
    st.write(f"{mydict[result]:.3f}" ,  unsafe_allow_html=True)


st.write("---")

cl1, cl2 = st.columns((2))

with cl1:
    plt.figure(figsize=(10, 8))
    st.subheader(f"{result} distribution")
    fig = sns.histplot(df[result], kde=True)
    st.pyplot(plt, clear_figure=True)

with cl2:
    # Count occurrences of each value
    value_counts = df['diagnosis'].value_counts()
    st.subheader("Class Distribution:")
    # Create a pie chart using Seaborn's pieplot function
    plt.figure(figsize=(4, 4))
    sns.set_theme(style="darkgrid")
    fig1 = plt.pie(value_counts, labels=["Benign","Maligant"], autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
    st.pyplot(plt, clear_figure=True)
