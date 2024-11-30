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
   dsqdqs
   
