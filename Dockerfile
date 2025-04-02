# Use Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy contents into container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the model is trained before starting the API
RUN python train.py

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
