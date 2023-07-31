import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

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
	selected = option_menu("Main Menu", ['Information', 'Exploratory Analysis', 'Machine Learning'])
	selected

df = pd.read_csv('./dataset_factorised.csv')
#divide the data into 2 classes
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

if 'Information' in selected:
	st.markdown("Features are computed from a digitized image of a fine needle \
            aspirate (FNA) of a breast mass. They describe characteristics of \
            the cell nuclei present in the image. n the 3-dimensional space is \
            that described in: [K. P. Bennett and O. L. Mangasarian: \"Robust Linear \
            Programming Discrimination of Two Linearly Inseparable Sets\", Optimization \
            Methods and Software 1, 1992, 23-34]. This database is also available through \
            the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/Also \
	    can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29")

if 'Exploratory Analysis' in selected:
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


	


#fig, ax = plt.subplots()
#sns.heatmap(df_col.corr(), ax=ax)
#st.write(fig)







