# Use official python  
FROM python:3.12.12-slim

# Set working directory inside the container 
WORKDIR /app

# Copy only dependency file first
COPY requirements.txt .

# Install Python dependencies 
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire project into docker image 
COPY . .

# Explicitely copy the model 
# Note the destination has been changed to match inference.py path 
# We are copying the folder into the container image
COPY src/serving/model /app/src/serving/model


# Copy MLflow run (artifacts and metadata) to container (/app/model/)
COPY src/serving/model/967343358783369273/models/m-af17d0dbc49c4616b0d58dc541adfb0c/artifacts/model /app/model
COPY src/serving/model/967343358783369273/28820b9f995946eda2f99b09bba86b03/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/967343358783369273/28820b9f995946eda2f99b09bba86b03/artifacts/preprocessing.pkl /app/model/preprocessing.pkl


# making serving and app importable without src prefix 
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app/src

# Expose fastapi port 
EXPOSE 8000

# Run FastAPI using uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

