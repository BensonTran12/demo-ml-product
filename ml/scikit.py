# scikit.py

# Step 1: Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pickle
import os

# Path to save/load the trained model
model_path = "trained_model.pkl"

# Step 2: Load dataset
titanic = sns.load_dataset('titanic')
print("✅ Data loaded successfully")
print(titanic.head())

# Step 3: Select features and clean data
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
titanic = titanic[features + ['survived']].dropna()

# Encode categorical variables
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked'], drop_first=True)

# Step 4: Split into train/test
X = titanic.drop('survived', axis=1)
y = titanic['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train or load model
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Loaded existing model")
else:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved")

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate model
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
