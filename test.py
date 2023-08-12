import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
#from pandasai import PandasAI
import openai
from pandasai.llm.openai import OpenAI

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression

st.set_page_config(
		page_title="Breast Cancer Predictor",
		page_icon=":female-doctor:",
		layout="wide",
		initial_sidebar_state="expanded"
	)

X_train_rfc = pd.read_csv('./data/X_train_rfc_feature_elim.csv', index_col=None)
X_test_rfc = pd.read_csv('./data/X_test_rfc_feature_elim.csv', index_col=None)
y_train = pd.read_csv('./data/y_train.csv', index_col=None)['diagnosis'] 
y_test = pd.read_csv('./data/y_test.csv',index_col=None)['diagnosis'] 

def add_sidebar():
	st.sidebar.header("Cell Nuclei Measurements")
	slider_labels = [
		("Radius (mean)", "radius_mean"),
		("Perimeter (mean)", "perimeter_mean"),
		("Concave points (mean)", "concave_points_mean"),
		("Area (se)", "area_se"),
		("Radius (worst)", "radius_worst"),
		("Area (worst)", "area_worst"),
		("Concave points (worst)", "concave_points_worst")
	]
	input_dict = {}
	for label, key in slider_labels:
		input_dict[key] = st.sidebar.slider(label, min_value=float(0),\
			max_value=float(X_test_rfc[key].max()), value=float(X_test_rfc[key].mean())
	)
	return input_dict

def get_radar_chart(input_data):
	categories = ['Radius', 'Perimeter', 'Area', 'Concave Points']
	fig = go.Figure()
	fig.add_trace(go.Scatterpolar(
		r=[X_train_rfc[['radius_mean', 'perimeter_mean', 'concave_points_mean']]],
		theta=categories,
		fill='toself',
		name='Mean Value'
	))
	fig.add_trace(go.Scatterpolar(
		r=[X_train_rfc['area_se']],
		theta=categories,
		fill='toself',
		name='Standard Error'
	))
	fig.add_trace(go.Scatterpolar(
		r=[X_train_rfc[['radius_worst', 'area_worst', 'concave_points_worst']]],
		theta=categories,
		fill='toself',
		name='Worst Value'
	))
	fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
	showlegend=True
	)
	return fig

def add_predictions(input_data):
	lr = LogisticRegression(solver='liblinear', random_state=12) 
	lr.fit(X_train_rfc, y_train)
	scaler = StandardScaler()  

	input_array = np.array(list(input_data.values())).reshape(1, -1)
	input_array_scaled = scaler.fit_transform(input_array)
	
	prediction = lr.predict(input_array_scaled)
	probabilities = lr.predict_proba(input_array_scaled)[0]	

	st.subheader("Cell cluster prediction")
	st.write("The cell cluster is:")
	
	if prediction[0] == 0:
		st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
	else:
		st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
	
	st.write("Probability of being benign: ", probabilities[0])
	st.write("Probability of being malicious: ", probabilities[1])
	
	st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
	return probabilities

def assistant(B, M):
	openai.api_key = "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"

	prompt = ("I build an app with Wisconsin breast cancer diagnosis and used \
		machine learning to give you these results, now I want you to be in \
		the role of assistant within that app and generate general guidelines \
		on what should he/she do when I give you the percentage now generate \
		guidelines for these predictions as you are talking to the patient:\n\
		Prediction Results:\nMalignant Probability: {M}\nBenign Probability: {B}")

	response = openai.Completion.create(
	model="text-davinci-003",
	prompt=prompt,
	temperature=0.6,
	max_tokens = 400
	)

	guidelines = response.choices[0].text.strip()
	return guidelines

def main():
	input_data = add_sidebar()	
	with st.container():
		st.title("Breast Cancer Predictor")
		st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
		col1, col2 = st.columns([4,1])
		with col1:
			radar_chart = get_radar_chart(input_data)
			st.plotly_chart(radar_chart)
			st.write("---")
		with col2:
			B , M = add_predictions(input_data)
			st.header("Ask the AI")
			st.write("Here you can ask the AI a question about the data")
			if st.button('Generate guidlines!'):
				with col1:
					guidelines = assistant(B, M)  # Call the assistant function and store the result
					st.write("Generated Guidelines:")  # Display a title for the generated guidelines
					st.write(guidelines) 

if __name__ == '__main__':
  main()