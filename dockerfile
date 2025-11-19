# Use official python  
FROM python:3.12.12-slim

# Set working directory inside the container 
WORKDIR /app

# Copy only dependency file first
COPY requirements.txt

# Install Python dependencies 
RUN pip install --upgrade pi \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire project into docker image 
COPY ..

# Explicitely copy the model 
# Nore destination has been changed to 
COPY src/serving/model /app/src/serving/model

