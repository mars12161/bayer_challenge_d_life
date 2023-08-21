import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle


def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
  )
  
  # train the model
  model = LogisticRegression(solver='liblinear', random_state = 12, stratify = y)
  model.fit(X_train, y_train)
  
  # test model
  y_pred = model.predict(X_test)
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(["Unnamed: 32", "id", "radius_mean", "texture_mean", 
                    "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                    "fractal_dimension_mean", "texture_se", "smoothness_se",
                    "compactness_se", "concavity_se", "concave points_se" , "symmetry_se", "fractal_dimension_se", "smoothness_worst",
                    "compactness_worst", "fractal_dimension_worst", "symmetry_mean"
                    ], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def main():
  data = get_clean_data()

  model, scaler = create_model(data)

  with open('./model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open('./model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()