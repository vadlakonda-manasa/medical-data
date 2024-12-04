import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:\Users\dharani\OneDrive\New folder\OneDrive\Desktop\data.csv")

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Check the shape of the DataFrame
print("Shape of the dataset:", data.shape)

# Check the columns
print("Columns in the dataset:", data.columns)

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Check for unique values in categorical columns
categorical_columns = ['Gender', 'Smoker', 'Diabetes', 'Heart Disease']
for column in categorical_columns:
    if column in data.columns:
        print(f"Unique values in '{column}': {data[column].unique()}")

# Replace inconsistent values
data['Smoker'] = data['Smoker'].replace({'Yes': 'Yes', 'No': 'No', 'yes': 'Yes', 'no': 'No', 'YES': 'Yes', 'NO': 'No'})
data['Diabetes'] = data['Diabetes'].replace({'Yes': 'Yes', 'No': 'No', 'yes': 'Yes', 'no': 'No', 'YES': 'Yes', 'NO': 'No'})
data['Heart Disease'] = data['Heart Disease'].replace({'Yes': 'Yes', 'No': 'No', 'yes': 'Yes', 'no': 'No', 'YES': 'Yes', 'NO': 'No'})

# Convert categorical variables to numerical ones
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

if 'Smoker' in data.columns:
    data['Smoker'] = data['Smoker'].map({'Yes': 1, 'No': 0})

if 'Diabetes' in data.columns:
    data['Diabetes'] = data['Diabetes'].map({'Yes': 1, 'No': 0})

if 'Heart Disease' in data.columns:
    data['Heart Disease'] = data['Heart Disease'].map({'Yes': 1, 'No': 0})

# Drop the 'Patient ID' column as it is not useful for prediction
if 'Patient ID' in data.columns:
    data.drop('Patient ID', axis=1, inplace=True)

# Splitting the dataset into features and target variable
X = data.drop('Heart Disease', axis=1)  # Features
y = data['Heart Disease']                # Target variable

# Check the data types of X and y
print("Data types of features:")
print(X.dtypes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')

# Show the plot
plt.show()
