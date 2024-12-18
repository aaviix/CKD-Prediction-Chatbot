# Chronic Kidney Disease Prediction and Chatbot Integration

This project integrates a machine learning-based Chronic Kidney Disease (CKD) prediction system with a Rasa-powered chatbot and a Streamlit frontend for user interaction. The chatbot provides CKD-related information, while the prediction system uses patient details to determine the likelihood of CKD.

## Features
- **CKD Prediction**: Logistic Regression and Decision Tree models are used to predict CKD based on patient input.
- **Chatbot Integration**: A Rasa chatbot answers user queries and guides them through the CKD prediction process.
- **Streamlit Interface**: A user-friendly web application to interact with the chatbot and input patient details for CKD prediction.
- **Visualization**: Interactive visualizations for CKD-related data analysis.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aaviix/ckd-prediction-chatbot.git
   cd ckd-prediction-chatbot
   ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ````

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ````

4. Train the Rasa bot:

    ```bash
    rasa train
    ````

---

## Usage

### Step 1: Start the Rasa Action Server

In the first terminal:

```bash
rasa run actions
```

### Step 2: Start the Rasa Server

In a second terminal:

```bash
rasa run
```

### Step 3: Run the Streamlit Application

In a third terminal:

```bash
streamlit run rasa-app.py
```

### Access the Application

Open your web browser and go to http://localhost:8501.

---

## Input Details

In the Streamlit application:

1. Provide patient details such as age, blood pressure, specific gravity, etc., in the input fields.

2. Click the "Predict" button to see predictions from Logistic Regression and Decision Tree models.

In the chatbot:

1. Ask about CKD (e.g., "What is CKD?").

2. Follow the chatbot's instructions for guidance or provide details for prediction.

---

## Data 

1. Dataset: The dataset used for predictions is Chronic-kidney-disease-final-edited.csv.

2. Preprocessing:
  - Missing values are filled with column means.
  - Categorical variables are one-hot encoded.
  - Features are scaled using StandardScaler.

---

## Model Used

- Logistic Regression:
    - A linear model trained to predict CKD.

- Decision Tree:
    - A non-linear model for classification based on feature splits.

---

## Visualizations

- Scatter Plot: Age vs. Blood Pressure.
- Line Plot: Age vs. Average Disease Classification.

---

## Future Improvements

- Add support for more machine learning models.
- Improve chatbot responses with additional CKD-related FAQs.
- Deploy the application on cloud platforms for public access.

---

## Requirements

The project requires the following dependencies (specified in requirements.txt):

- rasa
- streamlit
- pandas
- scikit-learn
- matplotlib

Install them using:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
