import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import RocCurveDisplay, auc
from sklearn import metrics

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
	selected = option_menu("Main Menu", ['Information', 'Exploratory Analysis', 'Machine Learning', 'Sources'])
	selected

df = pd.read_csv('./dataset_factorised.csv')
#divide the data into 2 classes
X = df.drop(['id','diagnosis'], axis = 1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state = 12)
Malignant=df[df['diagnosis'] == 0]
Benign=df[df['diagnosis'] == 1]

bins = 20

def histplot(features):
	plt.figure(figsize=(10,15))
	for i, feature in enumerate(features):
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

if 'Information' in selected:
	st.subheader('Information')
	st.markdown("An estimated 2.3 million females were diagnosed with breast cancer \
	    in 2020, accounting for approximately 24.5% of all cancer cases worldwide.  \n\
		Although female breast cancer incidence rates are lowest in less developed \
	    regions, mortality rates in these areas are similar to more \
	    developed countries due to lack of access to early detection and treatment.  \nThis \
	    project aims to improve the mass screening of populations and eventually decrease medical \
	    costs through computer-aided diagnosis. With decreased costs  \n")
	image1 = Image.open('images/figure2.png')
	st.image(image1)
	st.write("Source: https://canceratlas.cancer.org")    

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
	option_3 = st.selectbox('Please select a model:', ('Random Forest Classifier', 'Logistic Regression', 'Support Vector Machine', 'Ensemble Model'))
	if 'Random Forest Classifier' in option_3: 
		st.markdown("A **Random Forest Classifier** model was used with the following variables:  \nn_estimators = 40, max_depth = 4")
		rfc = RandomForestClassifier(n_estimators=40, max_depth=4) 
		rfc.fit(X_train, y_train)
		y_pred_rfc = rfc.predict(X_train)
		conf_rfc = confusion_matrix(y_train, y_pred_rfc)
		plot_heatmap(conf_rfc)
		st.markdown("The Random Forest Model has achieved:")
		results_rfc = pd.DataFrame({
    		'Score': ['accuracy', 'precision', 'recall', 'f1'], 
    '		Results': [rfc.score(X_train,y_train), precision_score(y_train, y_pred_rfc), recall_score(y_train, y_pred_rfc), f1_score(y_train, y_pred_rfc)]})
		st.write(results_rfc)
		conf_rfc = confusion_matrix(y_train, y_pred_rfc)
		st.write("**Confusion Matrix for Random Forest Classification Model**")
		st.pyplot(plot_heatmap(conf_rfc))
	if 'Logistic Regression' in option_3: 
		st.markdown("A **Logistic Regression** model was used with the following variables:  \nsolver='liblinear'")
		lr = LogisticRegression(solver='liblinear') 
		lr.fit(X_train,y_train)
		y_pred_lr = lr.predict(X_train)
		conf_lr = confusion_matrix(y_train, y_pred_lr)
		plot_heatmap(conf_lr)
		st.markdown("The Logistic Regression Model has achieved:")
		results_lr = pd.DataFrame({
    		'Score': ['accuracy', 'precision', 'recall', 'f1'], 
    '		Results': [lr.score(X_train,y_train), precision_score(y_train, y_pred_lr), recall_score(y_train, y_pred_lr), f1_score(y_train, y_pred_lr)]})
		st.write(results_lr)
		conf_lr = confusion_matrix(y_train, y_pred_lr)
		st.write("**Confusion Matrix for Logistic Regression Model**")
		st.pyplot(plot_heatmap(conf_lr))
	if 'Support Vector' in option_3: 
		st.markdown("A **Support Vector** model was used with the following variables:  \ndecision_function_shape='ovo', probability=True")
		svm = SVC(decision_function_shape='ovo', probability=True)
		svm.fit(X_train, y_train)
		y_pred_svm = svm.predict(X_train)
		conf_svm = confusion_matrix(y_train, y_pred_svm)
		plot_heatmap(conf_svm)
		st.markdown("The Support Vector Model has achieved:")
		results_svm = pd.DataFrame({
    		'Score': ['accuracy', 'precision', 'recall', 'f1'], 
    '		Results': [svm.score(X_train,y_train), precision_score(y_train, y_pred_svm), recall_score(y_train, y_pred_svm), f1_score(y_train, y_pred_svm)]})
		st.write(results_svm)
		conf_svm = confusion_matrix(y_train, y_pred_svm)
		st.write("**Confusion Matrix for Support Vector Machine**")
		st.pyplot(plot_heatmap(conf_svm))
	if 'Ensemble Model' in option_3: 
		st.markdown("An **Ensemble Model** was used with the following variables:  \nLogisticRegression(solver='liblinear'),  \nDecisionTreeClassifier,  \nSupport Vector Machine(kernel='rbf', probability=True)  \nand a Voting Classifier(voting='soft')")
		models = [('logreg', LogisticRegression(solver='liblinear')), ('tree', DecisionTreeClassifier()), ('svm', SVC(kernel='rbf', probability=True))]
		em = VotingClassifier(models, voting = 'soft')
		em.fit(X_train, y_train)
		#accuracy_em = em.score(X_train_em, y_train_em)
		y_pred_em = em.predict(X_train)
		st.markdown("The Ensemble Model has achieved:")
		results_em = pd.DataFrame({
    		'Score': ['accuracy', 'precision', 'recall', 'f1'], 
    '		Results': [em.score(X_train,y_train), precision_score(y_train, y_pred_em), recall_score(y_train, y_pred_em), f1_score(y_train, y_pred_em)]})
		st.write(results_em)
		conf_em = confusion_matrix(y_train, y_pred_em)
		st.write("**Confusion Matrix for Ensemble Model**")
		st.pyplot(plot_heatmap(conf_em))

if 'Sources' in selected:
	st.subheader('Sources')
	st.markdown("Dataset: http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  \nhttps://www.ncbi.nlm.nih.gov/pmc/articles/PMC8626596/  \n\
	     https://canceratlas.cancer.org/  \n")