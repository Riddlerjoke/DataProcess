import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Load and preprocess the dataset
data_path = os.path.join("data", "Sleep_health_and_lifestyle.xlsx")
df = pd.read_excel(data_path)

# Preprocessing (example based on previous description)
df.drop(columns=['Person ID'], inplace=True, errors='ignore')
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Good Sleep')
df.dropna(thresh=int(0.8 * len(df)), axis=1, inplace=True)
df['BMI Category'].replace(
    {"Norm": "Normal Weight", "Norma": "Normal Weight", "Normal": "Normal Weight", "Nan": "Normal Weight"},
    inplace=True)

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
X = df.drop(columns=['Sleep Disorder'])
y = df['Sleep Disorder']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment and tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Sleep Health XGBoost Experiment")

with mlflow.start_run():
    # Train an XGBoost model
    xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=40, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics in MLflow
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.xgboost.log_model(xgb_model, "model")

print("Model training complete with MLflow tracking.")
