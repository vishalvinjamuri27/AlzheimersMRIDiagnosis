import streamlit as st

import pandas as pd


from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score



# input seed number
seed = 0
np.random.seed(seed)

# input data
data = pd.read_csv('CZJ.csv',
                 names = ["t-tau","p-tau","p/t-ratio","A-beta","a-syn","14-3-3","patients"])
y=data.patients.values
x_data=data.drop(["patients"], axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data) -np.min(x_data))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
print ('K-Nearest: ', classifier.score(x_test, y_test))

# Create the main page
def main():
  st.title('Creutzfeldt-Jakob Disease Diagnosis')
  st.sidebar.header('Input')
 
 ttau= st.text_input("t-tau level:")
 ptau= st.text_input("p-tau level:")
 ptrat= st.text_input("p/t-ratio:")
 abeta= st.text_input("A-beta level:")
 fourteen= st.text_input("14-3-3 level:")
new_point=[(ttau,ptau,ptrat,abeta,fourteen)]
prediction = classifier.predict(new_point)
st.sidebar.success(f'Prediction: {prediction}')

if __name__ == '__main__':
  main()
