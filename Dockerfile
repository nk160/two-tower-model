# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model code and FastAPI application
COPY app/ ./app/
COPY Version4.py .

# Create log directory
RUN mkdir -p /var/log/app

# Set environment variables
ENV WANDB_API_KEY=your_actual_key_here

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
