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

# Run the scripts
CMD python script/ingest.py && \
    python script/script_train.py && \
    python script/script_score.py && \
    python -m http.server 5000
