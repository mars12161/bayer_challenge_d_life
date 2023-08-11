import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import RocCurveDisplay, auc, plot_roc_curve, plot_precision_recall_curve
#from sklearn import metrics

st.title('Breast Cancer Dataset')

st.markdown(
	"""
	<style>
		[data-testid=stSidebar] [data-testid=stImage]{
			text-align: center;
			display: block;
			margin-left: auto;
			margin-right: auto;
			width: 100%;
		}
	</style>
	""", unsafe_allow_html=True
)

#with st.sidebar:
#	image = Image.open('images/bc_awareness.png')
#	st.image(image, width=100)
#	selected = option_menu("Menu", ['Information', 'Exploratory Analysis', 'Machine Learning', 'Sources'])
#	selected

cd_2018 = pd.read_csv('./cd_2018.csv') #for the map on info page
df = pd.read_csv('./dataset_factorised.csv')
#divide the data into 2 classes
X = df.drop(['id','diagnosis'], axis = 1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state = 12)
Malignant=df[df['diagnosis'] == 0]
Benign=df[df['diagnosis'] == 1]
X_train_fr_rfc = pd.read_csv('./X_train_rfc_feature_elim.csv')
X_test_fr_rfc = pd.read_csv('./X_test_rfc_feature_elim.csv')

def histplot(features):
	plt.figure(figsize=(10,15))
	for i, feature in enumerate(features):
		bins = 20
		plt.subplot(5, 2, i+1)
		sns.histplot(Malignant[feature], bins=bins, color='blue', alpha=0.6, label='Malignant');
		sns.histplot(Benign[feature], bins=bins, color='pink', alpha=0.5, label='Benign');
		plt.title(str(' Density Plot of: ')+str(feature))
		plt.xlabel(str(feature))
		plt.ylabel('Count')
		plt.legend(loc='upper right')
	plt.tight_layout()
	plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_heatmap(confusion):
	
	plt.figure(figsize=(4,3))
	sns.heatmap(confusion, xticklabels = np.unique(y), yticklabels = np.unique(y),
				cmap = 'RdPu', annot=True, fmt='g')
	# fmt is used to switch off scientific notation
	plt.xlabel('Predicted', fontsize=14)
	plt.ylabel('Actual', fontsize = 14)

def ml_model(model, X_train, y_train, X_test, y_test):
	model.fit(X_train, y_train)
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)
	conf_model = confusion_matrix(y_test, y_test_pred)
	results_model_test = pd.DataFrame({
		'Score': ['accuracy', 'precision', 'recall', 'f1'],
		'Results': [model.score(X_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred)]})
	st.subheader("Test Scores")
	#st.write(results_model_test)
	st.subheader("Confusion Matrix on Test Data")
	st.pyplot(plot_heatmap(conf_model))
	st.subheader("ROC Curve on Test Data")
	st.write("The ROC and AUC are run on the test data after the model has been trained.")
	plot_roc_curve(model, X_test, y_test)
	st.pyplot()
	st.subheader("Precision-Recall Curve on Test Data")
	plot_precision_recall_curve(model, X_test, y_test)
	st.pyplot()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Information**", "**Exploratory Analysis**", "**Machine Learning**", "**Predictions**", "**Sources**"])
with tab1:
	st.header('Information')
	st.markdown("An estimated 2.1 million people were diagnosed with breast cancer \
		in 2018 worldwide.  It is the second leading cause of death by cancer in females (leading \
		cause is lung cancer).  \n  \nBreast cancer incidence rates are lowest in less developed regions however their mortality \
		rates are similar to more developed regions.  This would indicate that it is due to less early \
		detection.  \n  \nThis project aims to improve the mass screening of populations and \
		and decreasing medical costs through computer-aided diagnosis.  In addition, early detection has \
		been correlated with a higher rate of survival.\n")
	image1 = Image.open('images/figure2.png')
	st.image(image1)
	st.write("Source: https://canceratlas.cancer.org")
	st.subheader('Breast Cancer Deaths in 2018')
	st.write("Included in the hover data below is the current number of diagnosed cases of breast cancer per 100 people, in both sexes and age-standardized")
	fig = px.choropleth(cd_2018,
					 locations = "code", 
					 color = "deaths", 
					 hover_name = "country", 
					 hover_data = ["diagnosed"],
					 color_continuous_scale = px.colors.sequential.Sunsetdark)
	st.plotly_chart(fig)
	
with tab2:
	st.header('Exploratory Analysis')
	#divide feature names into groups
	mean_features= ['radius_mean','texture_mean','perimeter_mean',\
				'area_mean','smoothness_mean','compactness_mean',\
				'concavity_mean','concave_points_mean','symmetry_mean',\
				'fractal_dimension_mean']
	error_features=['radius_se','texture_se','perimeter_se',\
				'area_se','smoothness_se','compactness_se',\
				'concavity_se','concave_points_se','symmetry_se',\
				'fractal_dimension_se']
	worst_features=['radius_worst','texture_worst','perimeter_worst',\
				'area_worst','smoothness_worst','compactness_worst',\
				'concavity_worst','concave_points_worst',\
				'symmetry_worst','fractal_dimension_worst']
	option = st.selectbox(
		'What would you like to see?',
		('Density Graphs', 'Correlation or Heatmap'))

	if 'Density Graphs' in option: 
		option_1 = st.selectbox('Please select a group:', ('Mean Features', 'Standard Error Features', 'Worst Features'))
		if 'Mean Features' in option_1: 
			st.write(df[mean_features].describe())
			mf = histplot(mean_features)
			st.pyplot(mf)
		if 'Standard Error Features' in option_1: 
			st.write(df[error_features].describe())
			ef = histplot(error_features)
			st.pyplot(ef)
		if 'Worst Features' in option_1: 
			st.write(df[worst_features].describe())
			wf = histplot(worst_features)
			st.pyplot(wf)
	if 'Correlation or Heatmap' in option: 
		df_corr = df.drop(columns = ['id'])
		fig, ax = plt.subplots()
		option_2 = st.selectbox('Please select a group:', ('All', 'Mean Features', 'Standard Error Features', 'Worst Features'))
		if 'All' in option_2:
			sns.heatmap(df_corr.corr(), ax=ax)
			st.write(fig)
		if 'Mean Features' in option_2: 
			sns.heatmap(df_corr[mean_features].corr(), ax=ax)
			st.write(fig)
		if 'Standard Error Features' in option_2: 
			sns.heatmap(df_corr[error_features].corr(), ax=ax)
			st.write(fig)
		if 'Worst Features' in option_2: 
			sns.heatmap(df_corr[worst_features].corr(), ax=ax)
			st.write(fig)
with tab3:
	st.header('Machine Learning')
	st.write("All machine learning models were trained using an 80/20 split and using stratification.")
	option_3 = st.selectbox('Please select a model:', ('Random Forest Classifier', 'Logistic Regression', 'Support Vector Machine', 'Ensemble Model'))
	if 'Random Forest Classifier' in option_3: 
		st.subheader("Random Forest Classifier (or RFC)")
		st.write("Training Score Results before any feature elimination was: \n\
				accuracy: 1.0, precision: 1.0, recall: 1.0, and f1: 1.0\n")
		st.write("We reviewed the feature importance in SKlearn. We then eliminated all \
			features that had a coefficient value less than 0.025 and reran the RFC model to compare the outcome.")
		image = Image.open('images/rfc_features.png')
		st.image(image, width=100)
		st.write(results_model_test)
		st.markdown("A **Random Forest Classifier** model was used with the following variables: \n\
				n_estimators = 8, max_depth = 12 \nwhich was chosen due to the results of a GridSearch \
				Hyperparameter Optimization model using precision as a scoring metric on the full dataset.")
		rfc = RandomForestClassifier(n_estimators=8, max_depth=12)
		ml_model(rfc, X_train_fr_rfc, y_train, X_test_fr_rfc, y_test)
	if 'Logistic Regression' in option_3: 
		st.markdown("A **Logistic Regression** model was used with the following variables:  \nsolver='liblinear'")
		lr = LogisticRegression(solver='liblinear', random_state = 12) 
		ml_model(lr, X_train, y_train, X_test, y_test)
	if 'Support Vector' in option_3: 
		st.markdown("A **Support Vector** model was used with the following variables:  \ndecision_function_shape='ovo', probability=True")
		svm = SVC(decision_function_shape='ovo', probability=True)
		ml_model(svm, X_train, y_train, X_test, y_test)
	if 'Ensemble Model' in option_3: 
		st.markdown("An **Ensemble Model** was used with the following variables:  \nLogisticRegression(solver='liblinear'),  \nDecisionTreeClassifier,  \nSupport Vector Machine(kernel='rbf', probability=True)  \nand a Voting Classifier(voting='soft')")
		models = [('logreg', LogisticRegression(solver='liblinear')), ('tree', DecisionTreeClassifier()), ('svm', SVC(kernel='rbf', probability=True))]
		em = VotingClassifier(models, voting = 'soft')
		ml_model(em, X_train, y_train, X_test, y_test)
with tab4:
	st.subheader('Predictions')
	"""
	def get_clean_data():
		data = pd.read_csv("../data/data.csv")
		data = data.drop(['Unnamed: 32', 'id'], axis=1)
		data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
		return data

	def add_sidebar():
		st.sidebar.header("Cell Nuclei Measurements")

	data = get_clean_data()

	slider_labels = [
		("Radius (mean)", "radius_mean"),
		("Texture (mean)", "texture_mean"),
		("Perimeter (mean)", "perimeter_mean"),
		("Area (mean)", "area_mean"),
		("Smoothness (mean)", "smoothness_mean"),
		("Compactness (mean)", "compactness_mean"),
		("Concavity (mean)", "concavity_mean"),
		("Concave points (mean)", "concave points_mean"),
		("Symmetry (mean)", "symmetry_mean"),
		("Fractal dimension (mean)", "fractal_dimension_mean"),
		("Radius (se)", "radius_se"),
		("Texture (se)", "texture_se"),
		("Perimeter (se)", "perimeter_se"),
		("Area (se)", "area_se"),
		("Smoothness (se)", "smoothness_se"),
		("Compactness (se)", "compactness_se"),
		("Concavity (se)", "concavity_se"),
		("Concave points (se)", "concave points_se"),
		("Symmetry (se)", "symmetry_se"),
		("Fractal dimension (se)", "fractal_dimension_se"),
		("Radius (worst)", "radius_worst"),
		("Texture (worst)", "texture_worst"),
		("Perimeter (worst)", "perimeter_worst"),
		("Area (worst)", "area_worst"),
		("Smoothness (worst)", "smoothness_worst"),
		("Compactness (worst)", "compactness_worst"),
		("Concavity (worst)", "concavity_worst"),
		("Concave points (worst)", "concave points_worst"),
		("Symmetry (worst)", "symmetry_worst"),
		("Fractal dimension (worst)", "fractal_dimension_worst"),
	]

	input_dict = {}

	for label, key in slider_labels:
		input_dict[key] = st.sidebar.slider(label,
			min_value=float(0),
			max_value=float(data[key].max()),
			value=float(data[key].mean())
		)
	return input_dict

	def get_scaled_values(input_dict):
		data = get_clean_data()
		X = data.drop(['diagnosis'], axis=1)
		scaled_dict = {}
		for key, value in input_dict.items():
			max_val = X[key].max()
			min_val = X[key].min()
			scaled_value = (value - min_val) / (max_val - min_val)
			scaled_dict[key] = scaled_value
		return scaled_dict

	def get_radar_chart(input_data):
		input_data = get_scaled_values(input_data)
		categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
				'Smoothness', 'Compactness', 
				'Concavity', 'Concave Points',
				'Symmetry', 'Fractal Dimension']
		fig = go.Figure()
		fig.add_trace(go.Scatterpolar(
			r=[
				input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
				input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
				input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
				input_data['fractal_dimension_mean']
			],
			theta=categories,
			fill='toself',
			name='Mean Value'
		))
		fig.add_trace(go.Scatterpolar(
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
		))
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
#		model_lr = LogisticRegression(solver='liblinear', random_state = 12)
		model_p = pickle.load(open("../model/model.pkl", "rb"))
		scaler = pickle.load(open("../model/scaler.pkl", "rb"))
		input_array = np.array(list(input_data.values())).reshape(1, -1)
		input_array_scaled = scaler.transform(input_array)
		prediction = model_p.predict(input_array_scaled)
		st.subheader("Cell cluster prediction")
		st.write("The cell cluster is:")
		if prediction[0] == 0:
			st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
		else:
			st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
		st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
		st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
		st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
		return (model_p.predict_proba(input_array_scaled)[0][0], model_p.predict_proba(input_array_scaled)[0][1])

	def assistant(B, M):
		openai.api_key = "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"
		prompt = f"""

#	We built an app with the Wisconsin breast cancer dataset and used machine learning to provide these results, now we want you to be in the role of assistant within that app and generate general guidelines on what they should do when we give you the percentage.
#	Now generate guidelines for these predictions as if you are talking to the patient:

#	Prediction Results:
#	Malignant Probability: {M}
#	Benign Probability: {B}


		response = openai.Completion.create(
		model="text-davinci-003",
		prompt=prompt,
		temperature=0.6,
		max_tokens = 400
		)

		guidelines = response.choices[0].text.strip()
		return(guidelines)

	def main():
		st.set_page_config(
		page_title="Breast Cancer Predictor",
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
		if st.button('Generate guidelines!'):
			with col1:
				st.write(assistant(B, M))

with tab5:
	st.subheader('Dataset')
	st.markdown("http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic")
	st.subheader('Sources')
	st.markdown("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8626596/,  \n\
		 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7950292/,\n\
		 https://canceratlas.cancer.org/,  \nhttps://ourworldindata.org/cancer  \n")

if __name__ == '__main__':
	main()
