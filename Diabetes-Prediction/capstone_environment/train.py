import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# Define categorical and numerical features
categorical = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
numerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# XGBoost parameters
xgb_params = {
    'eta': 0.1, 
    'max_depth': 4,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Load and preprocess data
df = pd.read_csv(r"C:\Users\Aron\Desktop\Diabetes-Prediction\data\diabetes_prediction_dataset.csv")

# Map integer values to more descriptive categories
diabetes_values = {0: 'No', 1: 'Yes'}
df['diabetes'] = df['diabetes'].map(diabetes_values)

hypertension_values = {0: 'No Hypertension', 1: 'Has Hypertension'}
df['hypertension'] = df['hypertension'].map(hypertension_values)

heart_disease_values = {0: 'No Heart Disease', 1: 'Has Heart Disease'}
df['heart_disease'] = df['heart_disease'].map(heart_disease_values)

# Convert categorical columns to lowercase and replace spaces with underscores
string_columns = list(df.dtypes[df.dtypes == object].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Convert target column to binary
df['diabetes'] = (df['diabetes'] == 'yes').astype(int)

# Split data into training and testing sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_train = df_full_train['diabetes'].values
y_test = df_test['diabetes'].values

# Remove target column from the feature set
del df_full_train['diabetes']
del df_test['diabetes']

# Training
def train(df_train, y_train, num_boost_round=175, params=xgb_params):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(dv.get_feature_names_out()))
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return dv, model

# Prediction
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    y_pred = model.predict(dmatrix)
    return y_pred

# Train the final model
dv, model = train(df_full_train, y_train, num_boost_round=175, params=xgb_params)

# Predict on validation set and evaluate
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.3f}")

# Save the model and DictVectorizer to files
with open("dv.pkl", "wb") as f_out:
    pickle.dump(dv, f_out)

with open("xgb_model.pkl", "wb") as f_out:
    pickle.dump(model, f_out)

print("Model and vectorizer saved successfully.")
