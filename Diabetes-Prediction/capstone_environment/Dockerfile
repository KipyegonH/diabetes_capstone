# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to install dependencies
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY predict.py .
COPY dv.pkl .
COPY xgb_model.pkl .

EXPOSE 9696

# Specify the command to run the script
CMD ["python", "predict.py"]
