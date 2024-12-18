from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os

# Load and preprocess the dataset for CKD prediction
DATA_PATH = "Chronic-kidney-disease-final-edited.csv"  # Ensure this path is correct
data = pd.read_csv(DATA_PATH)

# Preprocessing
X = data.drop(columns='classification', axis=1)
Y = data['classification'].map({'ckd': 1, 'notckd': 0})

# Remove rows with NaN in target variable
valid_indices = Y.dropna().index
X = X.loc[valid_indices]
Y = Y.loc[valid_indices]

# One-hot encoding and handling missing values
X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
log_model = LogisticRegression(max_iter=7823)
log_model.fit(X_scaled, Y)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_scaled, Y)


class ActionCKDPrediction(Action):
    def name(self) -> Text:
        return "action_predict_ckd"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract slot values
        age = tracker.get_slot("age")
        bp = tracker.get_slot("bp")
        sg = tracker.get_slot("sg")
        al = tracker.get_slot("al")
        su = tracker.get_slot("su")
        rbc = tracker.get_slot("rbc")
        pc = tracker.get_slot("pc")
        pcc = tracker.get_slot("pcc")
        ba = tracker.get_slot("ba")

        # Create a DataFrame for prediction
        user_input = pd.DataFrame({
            'age': [float(age)],
            'bp': [float(bp)],
            'sg': [float(sg)],
            'al': [float(al)],
            'su': [float(su)],
            'rbc_normal': [1 if rbc == "normal" else 0],
            'pc_normal': [1 if pc == "normal" else 0],
            'pcc_present': [1 if pcc == "present" else 0],
            'ba_present': [1 if ba == "present" else 0]
        })

        # Match columns with training data
        for col in X.columns:
            if col not in user_input.columns:
                user_input[col] = 0

        # Reorder columns
        user_input = user_input[X.columns]

        # Scale the input
        user_input_scaled = scaler.transform(user_input)

        # Predict with both models
        log_result = log_model.predict(user_input_scaled)
        tree_result = tree_model.predict(user_input_scaled)

        # Determine prediction
        log_prediction = "CKD Detected" if log_result[0] == 1 else "No CKD Detected"
        tree_prediction = "CKD Detected" if tree_result[0] == 1 else "No CKD Detected"

        # Respond with results
        response = f"Logistic Regression Prediction: {log_prediction}\nDecision Tree Prediction: {tree_prediction}"
        dispatcher.utter_message(text=response)

        return []


class ActionExplainCKD(Action):
    def name(self) -> Text:
        return "action_explain_ckd"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Explanation of CKD
        response = (
            "Chronic Kidney Disease (CKD) is a condition characterized by gradual loss of kidney function. "
            "It often has no early symptoms but can be detected through blood tests, urine tests, or symptoms like "
            "high blood pressure, swelling, or fatigue. Consult a healthcare professional for accurate diagnosis and treatment."
        )
        dispatcher.utter_message(text=response)
        return []
