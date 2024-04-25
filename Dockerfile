# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Install Python dependencies
RUN pip install -r requirements.txt

# Run MLflow server in the background
CMD mlflow server --h 0.0.0.0 & \
    sleep 5 && \
    mlflow ui --port 5000 & \
    python script/main.py && \
    tail -f /dev/null
