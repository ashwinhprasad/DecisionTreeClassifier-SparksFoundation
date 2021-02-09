import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# header
st.markdown("""
    # Streamlit Decision Trees
    This App is a streamlit app for performing a simple classification task
    using random forest and to check which parameters give us an optimized
    result
""")

# display data
data = load_iris(as_frame=True)
x = data['data']
y = data['target']
st.markdown("""## Sample Data""")
st.write(x.head(),y.head())

# parameter
st.sidebar.header("Parameters")
n_estimators = st.sidebar.slider("n_estimators",1,200)

# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)

# training
clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(x_train,y_train)
y_pred  = clf.predict(x_test)
acc = (y_pred == y_test).mean()*100
st.write(f"The Accuracy of the model is {acc:.2f}%")

# making plot
pca = PCA(2)
x_test_d = pca.fit_transform(x_test)

fig = plt.figure()
plt.scatter(x_test_d[:,0],x_test_d[:,1],c=y_pred)
st.pyplot(fig)
