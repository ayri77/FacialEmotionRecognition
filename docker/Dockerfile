# TensorFlow GPU Docker Image for Facial Emotion Recognition
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter ipykernel ipython

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
