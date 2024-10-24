# Using python 3.12.7 slim image as the base image
FROM python:3.12.7-slim
# Defining working directory (within the container)
WORKDIR /usr/src/myapp
# Copy requirements.txt to the working directory
COPY requirements.txt .
# Install required python packages
RUN pip install --no-cache-dir -r requirements.txt
# Copy all files in prediction-ui local host directory to /usr/src/myapp in Docker container
COPY . .
# Expose the port that our Flask app will run on
EXPOSE 5000
# Environment Variables (update the API URL)
ENV PREDICTOR_API=http://predictor-api-server:5000/heart_attack_predictor/
# Metadata labels (optional)
LABEL heart.disease.predictor.api="0.0.1-beta"
# Run our Flask App
CMD ["python3", "app.py"]
