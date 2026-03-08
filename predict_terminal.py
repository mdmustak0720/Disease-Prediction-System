import pandas as pd
from joblib import load

# load trained model
model = load("./saved_model/decision_tree.joblib")

# load dataset to get symptom names
df = pd.read_csv("./dataset/training_data.csv")
symptoms = list(df.columns[:-2])   # first 132 columns are symptoms

print("\nAvailable Symptoms:")
for s in symptoms:
    print("-", s)

print("\nType symptoms separated by comma")
user_input = input("Enter symptoms: ")

input_symptoms = user_input.split(",")

# create input vector
input_data = [0] * len(symptoms)

for symptom in input_symptoms:
    symptom = symptom.strip()
    if symptom in symptoms:
        index = symptoms.index(symptom)
        input_data[index] = 1
    else:
        print(f"Symptom '{symptom}' not found in dataset")

# convert to dataframe
input_df = pd.DataFrame([input_data], columns=symptoms)

# predict disease
prediction = model.predict(input_df)

print("\nPredicted Disease:", prediction[0])