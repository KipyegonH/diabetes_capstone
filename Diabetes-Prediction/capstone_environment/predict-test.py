import requests

patient = {
 "gender": "female",
 "age": 30.0,
 "hypertension": "no_hypertension",
 "heart_disease": "has_heart_disease",
 "smoking_history": "current",
 "bmi": 30.73,
 "HbA1c_level": 6.6,
 "blood_glucose_level": 150
}


url = 'http://13.61.3.131:9696/predict'

response = requests.post(url, json=patient)
result = response.json()
print(result)
if result['diabetic'] == True:
    print('Patient has diabetes')
else:
    print('Patient is not diabetic')