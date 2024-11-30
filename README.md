# Diabetes Detection App

This project is a Python-based GUI application to detect diabetes using a custom logistic regression model. The application reads healthcare data, trains a machine learning model, and allows users to input patient data to predict the likelihood of diabetes.

## Features
- **Custom Logistic Regression Model**: Implements logistic regression from scratch.
- **GUI Interface**: User-friendly interface created with PyQt5 and Qt Designer.
- **Data Normalization**: Scales input data using `MinMaxScaler` to ensure accurate predictions.
- **Live Predictions**: Users can input data to get immediate predictions about diabetes.

## Dataset
The application uses a dataset named `Healthcare-Diabetes.csv`, which should have the following columns:
1. `Pregnancies`
2. `Glucose`
3. `BloodPressure`
4. `SkinThickness`
5. `Insulin`
6. `BMI`
7. `DiabetesPedigreeFunction`
8. `Age`
9. `Outcome` (binary: `0` for no diabetes, `1` for diabetes)

Make sure the dataset is in the same directory as the application.

## Installation 
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/diabetes-detection-app.git
   ```
2. Install dependencies:
   ```bash
   python pip install -m PyQt5 pandas numpy scikit-learn
   ```
   
## How to run 

1. Open a terminal in the project directory
2. Run the application:
   ```bash 
   python app.py
   ```
3. The GUI window will open. Input patient data and click the "Predict" button to detect diabetes.

## File Descriptions
 - `app.py`: Main application file containing the GUI and logic for the diabetes detection model.
 - `diabetes.ui`: The Qt Designer file defining the graphical interface.
 - `Healthcare-Diabetes.csv`: Dataset used for training and testing the model.
 - `README.md`:Documentation for the project.

## How it works
1. **Data Preprocessing**
   - The dataset is split into training (80%) and testing (20%) sets.
   - he features are normalized using `MinMaxScaler` for better model performance.
2. **Model Training**
   - A logistic regression model is implemented and trained from scratch.
   - The training process minimizes the binary cross-entropy loss.
3. **Prediction**
   - Input features are normalized using the same scaler as the training data.
   - The logistic regression model predicts the probability of diabetes.
4. **GUI**
   - The user inputs values for features like glucose levels, blood pressure, and BMI.
   - The application displays whether the patient is likely to have diabetes.

## EXAMPLE 
1. **Input**
 - Pregnancies: `5`
 - Glucose: `140`
 - Blood Pressure: `80`
 - Skin Thickness: `20`
 - Insulin: `100`
 - BMI: `28.5`
 - Diabetes Pedigree Function: `0.5`
 - Age: `45`

2. **Output**

  - `rak 3andk el sokkker` (likely diabetic)
  - or `salekt'ha` (not diabetic)

## Auther
   **Hakim Saoud**
   **50655hakim@gmail.com**
   **https://www.linkedin.com/in/hakim-saoud/**

