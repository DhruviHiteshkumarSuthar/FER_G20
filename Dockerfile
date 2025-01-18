# Use an official Python 3.10 image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including ffmpeg and other required libraries (for headless setup)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . /app

# Expose the port the app will run on
EXPOSE 8050

# Set the environment variable to ensure TensorFlow uses only CPU (no GPU)
ENV CUDA_VISIBLE_DEVICES=""

# Run the app
CMD ["python", "app.py"]