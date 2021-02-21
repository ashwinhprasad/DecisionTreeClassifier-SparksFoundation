# importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# header
st.image("./images/thumbnail.jpg")

st.markdown("""
    # Prediction Using Decision Tree
    This is a simple web app for performing a simple classification task
    using Decision Tree Algorithm and to check which parameters give us an optimized
    result
""")

# display data
data = pd.read_csv("./data/Iris.csv")
x = data.iloc[:,1:5]
y = data.iloc[:,5]

if st.button("Display Sample Data"):
    st.markdown("""## Sample Data""")
    st.write(x.head(),y.head())

# parameter
st.sidebar.header("Parameters")
criterion = st.sidebar.selectbox("Select Your Criterion",["gini","entropy"])

# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)

# fitting the model
clf = DecisionTreeClassifier(criterion=criterion)
clf.fit(x_train,y_train)
y_pred  = clf.predict(x_test)
acc = (y_pred == y_test).mean()*100
st.write(f"The Accuracy of the {criterion} based Decision Tree is {acc:.2f}%")

# vectorizing the prediction
le = LabelEncoder()
y_pred = le.fit_transform(y_pred)

# making plot
pca = PCA(2)
x_test_d = pca.fit_transform(x_test)
fig = plt.figure()
plt.scatter(x_test_d[:,0],x_test_d[:,1],c=y_pred)
st.pyplot(fig)

# decision tree plot
st.write("## Tree Visualization ")
fig = plt.figure(figsize=(5,10),dpi=100)
plot_tree(clf,filled=True)
st.pyplot(fig)

# new prediction
new_sepalLength = st.number_input(label="Sepal Length")
new_sepalWidth = st.number_input(label="Sepal Width")
new_petalLength = st.number_input(label="Petal Length")
new_petalWidth = st.number_input(label="Petal Width")

#make prediction
x_test = np.array([new_petalLength,new_sepalWidth,new_petalLength,new_petalWidth])
st.write(clf.predict(x_test.reshape(-1,4)))


# github directory for code
st.markdown("""
    Github Repo : https://github.com/ashwinhprasad/DecisionTreeClassifier-SparksFoundation
""")
