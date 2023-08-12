import streamlit as st
#import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
#from pandasai import PandasAI
#import openai
#from pandasai.llm.openai import OpenAI

X_train_lr = pd.read_csv('./data/X_train_lr.csv')
X_test_lr = pd.read_csv('./data/X_test_lr.csv')

def add_sidebar():
	st.sidebar.header("Cell Nuclei Measurements")
	slider_labels = [
		("Radius (mean)", "radius_mean"),
		("Texture (mean)", "texture_mean"),
		("Perimeter (mean)", "perimeter_mean"),
		("Area (mean)", "area_mean"),
		("Smoothness (mean)", "smoothness_mean"),
		("Compactness (mean)", "compactness_mean"),
		("Concavity (mean)", "concavity_mean"),
		("Concave points (mean)", "concave points_mean"),
		("Radius (se)", "radius_se"),
		("Texture (se)", "texture_se"),
		("Perimeter (se)", "perimeter_se"),
		("Area (se)", "area_se"),
		("Compactness (se)", "compactness_se"),
		("Symmetry (se)", "symmetry_se"),
		("Fractal dimension (se)", "fractal_dimension_se"),
		("Radius (worst)", "radius_worst"),
		("Texture (worst)", "texture_worst"),
		("Perimeter (worst)", "perimeter_worst"),
		("Area (worst)", "area_worst"),
		("Smoothness (worst)", "smoothness_worst"),
		("Concavity (worst)", "concavity_worst"),
		("Concave points (worst)", "concave points_worst"),
		("Symmetry (worst)", "symmetry_worst"),
		("Fractal dimension (worst)", "fractal_dimension_worst"),
	]

	input_dict = {}

	for label, key in slider_labels:
		input_dict[key] = st.sidebar.slider(label, min_value=float(0),\
			max_value=float(X_test_lr[key].max()), value=float(X_test_lr[key].mean())
	)

def get_radar_chart(input_data):
  
  input_data = X_test_lr
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
				'Smoothness', 'Compactness', 
				'Concavity', 'Concave Points',
				'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
		r=X_test_lr,
		theta=categories,
		fill='toself',
		name='Mean Value'
  ))
  """fig.add_trace(go.Scatterpolar(
		r=[
		  input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
		  input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
		  input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
		],
		theta=categories,
		fill='toself',
		name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
		r=[
		  input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
		  input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
		  input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
		  input_data['fractal_dimension_worst']
		],
		theta=categories,
		fill='toself',
		name='Worst Value'
  ))"""

  fig.update_layout(
	polar=dict(
	  radialaxis=dict(
		visible=True,
		range=[0, 1]
	  )),
	showlegend=True
  )
  
  return fig


def add_predictions(input_data):
	lr = LogisticRegression(solver='liblinear', random_state = 12) 
	lr.fit(X_train_lr, y_train)
	scaler = StandardScaler()  
	input_array = np.array(list(input_data.values())).reshape(1, -1)
	input_array_scaled = scaler.transform(input_array)
	prediction = model.predict(input_array_scaled)
	st.subheader("Cell cluster prediction")
	st.write("The cell cluster is:")

	if prediction[0] == 0:
		st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
	else:
		st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
	
	st.write("Probability of being benign: ", lr.predict_proba(input_array_scaled)[0][0])
	st.write("Probability of being malicious: ", lr.predict_proba(input_array_scaled)[0][1])

	st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
	#return (lr.predict_proba(input_array_scaled)[0][0], lr.predict_proba(input_array_scaled)[0][1])

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


def main():
  st.set_page_config(
	#page_title="Breast Cancer Predictor",
	page_icon=":female-doctor:",
	layout="wide",
	initial_sidebar_state="expanded"
	)

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