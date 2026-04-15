FROM python:3.11-slim

WORKDIR /app

COPY scheduling-shared /shared
RUN pip install --no-cache-dir /shared

COPY scheduling-worker-api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY scheduling-worker-api/ .
COPY scheduling /app/scheduling

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]

