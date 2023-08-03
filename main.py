import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import streamlit as st


# def get_xy(data:pd.DataFrame,list_drp:list):
#     """
#     set the x and y column
    
#     args:
#         data(pd.DataFrame): the dataFrame which we are extracting the x and y
    
#     returns:
#         y and X in form of pandas series
    
#     """
#     y = data.diagnosis # M or B 
#     X = data.drop(list_drp,axis = 1 )
#     return y,X


# def display_pie_chart(data_series):
#     # Count occurrences of each value
#     value_counts = data_series.value_counts()

#     # Create a pie chart using Seaborn's pieplot function
#     plt.figure(figsize=(6, 6))
#     plt.pie(value_counts, labels=["Benign","Maligant"], autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
#     plt.title('Pie Chart of Value Distribution')
    
#     # Show the pie chart
#     plt.show()


# def main():
#     st.title("Pie Chart with Streamlit")

#     # Display the pie chart
#     display_pie_chart(y)

def display_corr_bar(data_frame):

    data_frame['diagnosis'] = data_frame['diagnosis'].map({'M': 1, 'B': 0})
    corr_diagnosis = data_frame.corr()['diagnosis'].sort_values(ascending=False)
    corr_diagnosis3 = corr_diagnosis[1:5]
# Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

# Create a seaborn bar plot
    ax = sns.barplot(x=corr_diagnosis3.values, y=corr_diagnosis3.index, palette="rocket")
    ax.bar_label(ax.containers[0])

# Add labels and title
    plt.title('Correlation of Features with Diagnosis', fontsize = 40, pad = 20)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    return plt


def display_pie_chart(data_series):
    # Count occurrences of each value
    value_counts = data_series.value_counts()

    # Create a pie chart using Seaborn's pieplot function
    plt.figure(figsize=(4, 4))
    sns.set_theme(style="whitegrid")
    plt.pie(value_counts, labels=["Benign","Maligant"], autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
    #plt.title('Pie Chart of Value Distribution')

    # Return the Matplotlib figure
    return plt

def main():
    df = pd.read_csv("data.csv")

    st.set_page_config(layout='wide', initial_sidebar_state='expanded')
    st.title(":bar_chart: Minimal dashboard team d-life")
    st.header("A quick overview of breast cancer dataset(diagnosis)")


    df = df.drop(columns=["id", "Unnamed: 32"])

    c1, c2 = st.columns((3,2))
    with c1:
        st.pyplot(display_corr_bar(df))
    with c2:
        st.pyplot(display_pie_chart(df["diagnosis"]))
    st.write("---")

    
    corr_diagnosis = df.corr()['diagnosis'].sort_values(ascending=False)
    mydict = corr_diagnosis.to_dict()
    result = st.selectbox("Select your feature:",list(mydict.keys()))
    min_result = df[result].min()
    max_result = df[result].max()
    initial_value = [float(min_result), float(max_result)]
    sel_range = st.slider("The range", min_value=float(min_result), max_value=float(max_result), value=initial_value)
    lower_bound, upper_bound = sel_range
    filtered_df = df[(df[result] >= lower_bound) & (df[result] <= upper_bound)]
    st.write(f"the correlation is: {mydict[result]} ")
    st.pyplot(display_pie_chart(filtered_df["diagnosis"]))

    

if __name__ == "__main__":
    main()



    

# st.sidebar.header('Dashboard `version 2`')

# st.sidebar.subheader('Heat map parameter')
# time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 

# st.sidebar.subheader('Donut chart parameter')
# donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

# st.sidebar.subheader('Line chart parameters')
# plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
# plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

# st.sidebar.markdown('''
# ---
# Created with ❤️ by [Data Professor](https://youtube.com/dataprofessor/).
# ''')
