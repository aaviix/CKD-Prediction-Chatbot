import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Read the dataset
data = pd.read_csv('Chronic-kidney-disease-final-edited.csv')

# Data preprocessing
X = data.drop(columns='classification', axis=1)
Y = data['classification'].map({'ckd': 1, 'notckd': 0})

# Removing rows where target variable 'Y' is NaN
valid_indices = Y.dropna().index
X = X.loc[valid_indices]
Y = Y.loc[valid_indices]

# One-hot encode categorical variables and handle missing values
X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=2, stratify=Y)

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=7823)
log_model.fit(x_train, y_train)

# Train Decision Tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)

# Streamlit App
st.title("Chronic Kidney Disease Prediction")

# Sidebar for inputs
st.sidebar.header("Input Patient Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1, value=30)
bp = st.sidebar.number_input("Blood Pressure", min_value=50, max_value=180, step=1, value=70)
sg = st.sidebar.number_input("Specific Gravity", min_value=1.0, max_value=1.02, step=0.01, value=1.01)
al = st.sidebar.number_input("Albumin", min_value=0, max_value=5, step=1, value=2)
su = st.sidebar.number_input("Sugar", min_value=0, max_value=5, step=1, value=1)
rbc = st.sidebar.selectbox("Red Blood Count", options=["Normal", "Abnormal"])
pc = st.sidebar.selectbox("Pus Cell", options=["Normal", "Abnormal"])
pcc = st.sidebar.selectbox("Pus Cell Clumps", options=["Present", "Notpresent"])
ba = st.sidebar.selectbox("Bacteria", options=["Present", "Notpresent"])

# Prepare the user input as a DataFrame
user_input = pd.DataFrame({
    'age': [age], 'bp': [bp], 'sg': [sg], 'al': [al], 'su': [su],
    'rbc_normal': [1 if rbc == "Normal" else 0],
    'pc_normal': [1 if pc == "Normal" else 0],
    'pcc_present': [1 if pcc == "Present" else 0],
    'ba_present': [1 if ba == "Present" else 0]
})

# Match the input features to the training data format
for col in X.columns:
    if col not in user_input.columns:
        user_input[col] = 0

# Reorder columns to match training data
user_input = user_input[X.columns]

# Scale the input
user_input_scaled = scaler.transform(user_input)

# Prediction
if st.sidebar.button("Predict"):
    # Logistic Regression Prediction
    log_pred = log_model.predict(user_input_scaled)
    log_result = "Chronic Kidney Disease" if log_pred[0] == 1 else "No Chronic Kidney Disease"

    # Decision Tree Prediction
    tree_pred = tree_model.predict(user_input_scaled)
    tree_result = "Chronic Kidney Disease" if tree_pred[0] == 1 else "No Chronic Kidney Disease"

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Logistic Regression:** {log_result}")
    st.write(f"**Decision Tree Classifier:** {tree_result}")

# Visualizations
st.subheader("Data Analysis")
if st.checkbox("Show Scatter Plot of Age vs Blood Pressure"):
    plt.scatter(data['age'], data['bp'])
    plt.xlabel("Age")
    plt.ylabel("Blood Pressure")
    plt.title("Age vs Blood Pressure")
    st.pyplot(plt)

if st.checkbox("Show Line Plot of Age vs Average Disease Classification"):
    avg_classification = data.groupby('age')['classification'].apply(lambda x: (x == 'ckd').mean())
    plt.plot(avg_classification.index, avg_classification.values)
    plt.xlabel("Age")
    plt.ylabel("Average Disease Classification")
    plt.title("Age vs Average Disease Classification")
    st.pyplot(plt)
