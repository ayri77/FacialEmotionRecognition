#!/bin/bash

echo "🐳 Building Docker image for Facial Emotion Recognition App testing..."
echo "This will test the app in an environment similar to Streamlit Cloud"

# Build the image
docker-compose build

echo "🚀 Starting the application..."
echo "The app will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop"

# Start the container
docker-compose up

echo "🛑 Stopping containers..."
docker-compose down
