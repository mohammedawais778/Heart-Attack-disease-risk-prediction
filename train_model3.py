import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("HeartDataset.csv")

# Print column names for verification
print("Dataset Columns:", df.columns)

# Handle missing values (if any)
df.dropna(inplace=True)

# Define the 9 features used in `app2.py`
expected_features = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina"
]
target_column = "HeartDisease"

# Validate dataset columns
missing_columns = [col for col in expected_features + [target_column] if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}. Available columns: {df.columns}")

# Select only the required features
X = df[expected_features]
y = df[target_column]

# Convert categorical features to numeric (same as in `app2.py`)
X["Sex"] = X["Sex"].map({"Male": 1, "Female": 0})
X["FastingBS"] = X["FastingBS"].map({1: 1, 0: 0})
X["ExerciseAngina"] = X["ExerciseAngina"].map({"Yes": 1, "No": 0})

chest_pain_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
resting_ecg_dict = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}

X["ChestPainType"] = X["ChestPainType"].map(chest_pain_dict)
X["RestingECG"] = X["RestingECG"].map(resting_ecg_dict)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained with 9 features and saved as model.pkl")
