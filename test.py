import requests

url = "http://localhost:9696/predict"  # docker and FastAPI url

#url = " " # cloud url

#cloud_host_address = " "

# Original data frame index of clients for test
# 1:  6 female, 26 Male
# 0: 3 female, 4 Male

#client_male_diabetic
client = {
    'gender': 'Male',
     'age': 67.0,
     'hypertension': 0,
     'heart_disease': 1,
     'smoking_history': 'not_current',
     'bmi': 27.32,
     'HbA1c_level': 6.5,
     'blood_glucose_level': 200
}


#client_female_diabetic
client = {
    'gender': 'Female',
     'age': 44.0,
     'hypertension': 0,
     'heart_disease': 0,
     'smoking_history': 'never',
     'bmi': 19.31,
     'HbA1c_level': 6.5,
     'blood_glucose_level': 200
}

#client_male_nondiabetic
client = {
    'gender': 'Male',
     'age': 76.0,
     'hypertension': 1,
     'heart_disease': 1,
     'smoking_history': 'current',
     'bmi': 20.14,
     'HbA1c_level': 4.8,
     'blood_glucose_level': 155
}

#client_female_nondiabetic
client = {
    'gender': 'Female',
     'age': 36.0,
     'hypertension': 0,
     'heart_disease': 0,
     'smoking_history': 'current',
     'bmi': 23.45,
     'HbA1c_level': 5.0,
     'blood_glucose_level': 155
}

diagnosis = requests.post(url, json=client).json()
print("Probability of diagnosis = ", diagnosis)