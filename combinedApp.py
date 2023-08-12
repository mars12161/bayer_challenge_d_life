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

with st.sidebar:
	image = Image.open('images/bc_awareness.png')
	st.image(image, width=100)
	selected = option_menu("Menu", ['Information', 'Exploratory Analysis', 'Machine Learning', 'Predictions', 'Sources'])
	selected

cd_2018 = pd.read_csv('./data/cd_2018.csv') #data needed for map
df = pd.read_csv('./data/dataset_factorised.csv') #data needed for EDA
X_scaler = pd.read_csv('./data/X_scaler.csv') #StandardScaler data
X_train_fr_rfc = pd.read_csv('./data/X_train_rfc_feature_elim.csv') #RFC feature elim.
X_test_fr_rfc = pd.read_csv('./data/X_test_rfc_feature_elim.csv') #RFC feature elim.
X_train_lr = pd.read_csv('./data/X_train_lr.csv')
X_test_lr = pd.read_csv('./data/X_test_lr.csv')

y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X_scaler,y, test_size = .20, random_state = 12,  stratify = y)
Malignant=df[df['diagnosis'] == 0]
Benign=df[df['diagnosis'] == 1]

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
	results_model_test = pd.DataFrame({
		'Score': ['accuracy', 'precision', 'recall', 'f1'],
		'Results': [model.score(X_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred)]})
	st.subheader("Test Scores")
	st.write(results_model_test)

if 'Information' in selected:
	st.subheader('Information')
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
	
if 'Exploratory Analysis' in selected:
	st.subheader('Exploratory Analysis')
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
if 'Machine Learning' in selected:
	st.subheader('Machine Learning')
	st.write("All machine learning models were trained using an 80/20 split on stratified data that was standardised using StandardScaler.")
	st.subheader('ROC and AUC for All Models')
	image_all_ROC = Image.open('./images/All_Models_ROC.png')
	st.image(image_all_ROC)
#	st.subheader('Precision and Recall for All Models')
#	image_all_pr = Image.open('./images/All_Models_precisionrecall.png')
#	st.image(image_all_pr)
	option_3 = st.selectbox('**Please select a model you would like to explore further:**', ('Random Forest Classifier', 'Logistic Regression', 'Support Vector Machine', 'Ensemble Model'))
	if 'Random Forest Classifier' in option_3: 
		st.subheader("Random Forest Classifier (or RFC)")
		st.markdown("A **Random Forest Classifier** model was used with the following variables: \n\
			n_estimators = 4, max_depth = 8 \nwhich was chosen due to the results of a GridSearch \
			Hyperparameter Optimization model using precision as a scoring metric on the full dataset.")
		st.write("Training Score Results before any feature elimination was: \n\
			accuracy: 1.0, precision: 0.996, recall: 0.989, and F1: 0.992\n")
		st.write("We reviewed the feature importance in SKlearn. We then eliminated all \
			features that had a coefficient value less than 0.025 and reran the RFC model to compare the outcome.")
		image_rfc_feat = Image.open('./images/rfc_features.png')
		st.image(image_rfc_feat)
		st.subheader("Random Forest Classifier (or RFC) with Feature Elimination")
		rfc = RandomForestClassifier(n_estimators=4, max_depth=8, random_state = 12)
		ml_model(rfc, X_train_fr_rfc, y_train, X_test_fr_rfc, y_test)
		st.subheader("Confusion Matrix on Test Data")
		image_rfc = Image.open('./images/rfc_feature_elim_confusion_matrix.png')
		st.image(image_rfc)
		st.subheader("ROC Curve on Test Data")
		st.write("The ROC and AUC are run on the test data after the model has been trained.")
		image_rfc_ROC = Image.open('./images/rfc_ROC.png')
		st.image(image_rfc_ROC)
		st.subheader("Precision-Recall Curve on Test Data")
		image_rfc_precision = Image.open('./images/rfc_precision_recall.png')
		st.image(image_rfc_precision)
	if 'Logistic Regression' in option_3: 
		st.markdown("A **Logistic Regression** model was used with the following solver = \
			'liblinear'.")
		st.markdown("Training Score Results before any feature elimination was:\n\
			accuracy: 1.0, precision: 0.986, recall: 0.992, and F1: 0.999")
		st.markdown("We then reviewed the feature importance for the model and decided to \
			remove features that had a coefficient less than +/- 0.025")
		image_lr_feat = Image.open('./images/lr_features.png')
		st.image(image_lr_feat)
		lr = LogisticRegression(solver='liblinear', random_state = 12) 
		ml_model(lr, X_train_lr, y_train, X_test_lr, y_test)
		st.subheader("Confusion Matrix on Test Data")
		image_lr = Image.open('./images/lr_feature_elim_confusion_matrix.png')
		st.image(image_lr)
		st.subheader("ROC Curve on Test Data")
		st.write("The ROC and AUC are run on the test data after the model has been trained.")
		image_lr_ROC = Image.open('./images/lr_ROC.png')
		st.image(image_lr_ROC)
		st.subheader("Precision-Recall Curve on Test Data")
		image_lr_precision = Image.open('./images/lr_precision_recall.png')
		st.image(image_lr_precision)
	if 'Support Vector' in option_3: 
		st.markdown("A **Support Vector** model was used with the following parameters: decision_function_shape='ovo', probability=True \
	      and using the reduced features resulting from the logistic regression model.")
		st.markdown("Training Score Results before any feature elimination was:\n\
			accuracy: 1.0, precision: 0.982, recall: 0.996, and F1: 0.999")
		svm = SVC(decision_function_shape='ovo', probability=True)
		ml_model(svm, X_train_lr, y_train, X_test_lr, y_test)
		st.subheader("Confusion Matrix on Test Data")
		image_svm = Image.open('./images/svm_feature_elim_confusion_matrix.png')
		st.image(image_svm)
		st.subheader("ROC Curve on Test Data")
		st.write("The ROC and AUC are run on the test data after the model has been trained.")
		image_svm_ROC = Image.open('./images/svm_ROC.png')
		st.image(image_svm_ROC)
		st.subheader("Precision-Recall Curve on Test Data")
		image_svm_precision = Image.open('./images/svm_precision_recall.png')
		st.image(image_svm_precision)
	if 'Ensemble Model' in option_3: 
		st.markdown("An **Ensemble Model** was used with the following parameters: LogisticRegression(solver='liblinear'), DecisionTreeClassifier, Support Vector Machine(kernel='rbf', probability=True), and a Voting Classifier(voting='soft')")
		st.markdown("Training Score Results before any feature elimination was:\n\
			accuracy: 1.0, precision: 0.99, recall: 1.0, and F1: 0.995")
		models = [('logreg', LogisticRegression(solver='liblinear')), ('tree', DecisionTreeClassifier()), ('svm', SVC(kernel='rbf', probability=True))]
		em = VotingClassifier(models, voting = 'soft')
		ml_model(em, X_train_lr, y_train, X_test_lr, y_test)
		st.subheader("Confusion Matrix on Test Data")
		image_em = Image.open('./images/em_feature_elim_confusion_matrix.png')
		st.image(image_em)
		st.subheader("ROC Curve on Test Data")
		st.write("The ROC and AUC are run on the test data after the model has been trained.")
		image_em_ROC = Image.open('./images/em_ROC.png')
		st.image(image_em_ROC)
		st.subheader("Precision-Recall Curve on Test Data")
		image_em_precision = Image.open('./images/em_precision_recall.png')
		st.image(image_em_precision)

if 'Sources' in selected:
	st.subheader('Dataset')
	st.markdown("http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic")
	st.subheader('Sources')
	st.markdown("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8626596/,  \n\
		 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7950292/,\n\
		 https://canceratlas.cancer.org/,  \nhttps://ourworldindata.org/cancer  \n")

if 'Predictions' in selected:
	st.subheader('Predictions')