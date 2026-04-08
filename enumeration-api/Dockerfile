# Use Python 3.11 slim image as base
# Slim images are smaller and more secure than full images
FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands will run from this directory
WORKDIR /app

# Install shared package
COPY enumeration-shared /shared
RUN pip install --no-cache-dir /shared

# Copy requirements file first to leverage Docker layer caching
# If requirements.txt doesn't change, Docker will use cached layer
COPY enumeration-api/requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not storing pip cache
# --upgrade ensures pip is up to date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
# This is done after installing dependencies to optimize caching
COPY enumeration-api/ .

# Expose port 8000 for the FastAPI application
# This is the port Uvicorn will listen on
EXPOSE 8000

# Run the FastAPI application using Uvicorn ASGI server
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port to listen on
# --reload enables auto-reload on code changes (useful for development)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
