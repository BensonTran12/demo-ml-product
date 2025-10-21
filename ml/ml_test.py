import pickle
import pandas as pd

# Load trained model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Use it
new_data = pd.DataFrame([{
    "pclass": 1,
    "age": 38,
    "sibsp": 1,
    "parch": 0,
    "fare": 71.2833,
    "sex_male": 0,
    "embarked_Q": 0,
    "embarked_S": 1
}])
prediction = model.predict(new_data)
print("Survival prediction:", prediction)
