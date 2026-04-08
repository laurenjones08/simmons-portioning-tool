FROM python:3.11-slim

WORKDIR /app

# Install shared package
COPY enumeration-shared /shared
RUN pip install --no-cache-dir /shared

COPY enumeration-worker-api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY enumeration-worker-api/ .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

