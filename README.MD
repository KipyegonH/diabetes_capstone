# Diabetes Prediction Capstone

## Project Overview

This project applies machine learning techniques to predict the likelihood of diabetes based on health-related attributes. The aim is to develop and deploy a model that can process new input data and return accurate predictions.

---

## Dataset

The dataset, available at [UCI Repository](https://archive.ics.uci.edu/static/public/34/diabetes.zip), includes key health metrics for individuals. Attributes include:

- **Gender**: Male, Female, Other.
- **Age**: Ranges from 0 to 80 years.
- **BMI**: Categorized as Underweight, Normal, Overweight, or Obese.
- **Hypertension**: Binary indicator (0 = No, 1 = Yes).
- **Heart Disease**: Binary indicator (0 = No, 1 = Yes).
- **Smoking History**: Categories: Never, Former, Current, Not Current, Ever, No Info.
- **Blood Glucose Level**: Measure of glucose in the bloodstream.
- **HbA1c Level**: Average blood sugar level over 2-3 months.

**Target Variable:**

- **Diabetes**: Binary outcome (0 = No diabetes, 1 = Has diabetes).

---

## Project Structure

The repository includes the following:

- **README.md**: Project overview and setup instructions.
- **data/**: Dataset files.
- **notebook.ipynb**: Jupyter Notebook for data exploration, preprocessing, and modeling.
- **capstone\_environment/**: Deployment-related scripts:
  - `train.py`: Trains and saves the final model.
  - `predict.py`: Hosts the model as a web service.
  - `Dockerfile`: Defines the Docker container setup.
- **requirements.txt**: Required Python packages.
- **models/**: Contains saved models and preprocessing files:
  - `dv.pkl`: Preprocessing pipeline.
  - `xgb_model.pkl`: Trained XGBoost model.

---

## Installation and Usage

### Clone the Repository

```bash
git clone https://github.com/KipyegonH/diabetes_capstone.git
cd diabetes_capstone
```

### Build and Run with Docker

1. **Build Docker Image:**

   ```bash
   docker build -t diabetes-prediction .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 80:80 diabetes-prediction
   ```

3. **Test the API:**
   Use `curl` or Postman to send a POST request:

   ```bash
   curl -X POST -H "Content-Type: application/json" \
   -d '{"Gender": "Female", "Age": 45, "BMI": "Overweight", "Hypertension": 0, "Heart Disease": 1, "Smoking History": "Former", "Blood Glucose Level": 120, "HbA1c Level": 6.8}' \
   http://localhost/predict
   ```

   **Response:**

   ```json
   {
     "diabetes": 1
   }
   ```

---

## Deployment on AWS ECS (Optional)

Follow these steps to deploy your Docker container on AWS ECS:

1. Push your Docker image to Amazon Elastic Container Registry (ECR).
2. Create an ECS cluster and a task definition using your ECR image.
3. Set up a service in ECS to run the task.
4. (Optional) Use API Gateway to expose the service to the internet.

Refer to the [AWS ECS Documentation](https://aws.amazon.com/ecs/) for detailed steps.

---

## Contributors

[Kipyegon H.](https://github.com/KipyegonH)

