FROM python:3.10-slim
ARG ENV
ENV ENV=$ENV

ARG GCP_PROJECT_ID
ENV GCP_PROJECT_ID=$GCP_PROJECT_ID

ARG GCP_KEY

WORKDIR /app

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Decodificar el base64 y guardar como JSON
RUN mkdir -p deployment && \ 
    if [ -n "$GCP_KEY" ]; then \
        echo "$GCP_KEY" | base64 --decode > deployment/gcp_key.json && \
        echo "Config JSON created successfully"; \
    else \
        echo "No GCP_KEY provided, skipping gcp.json creation"; \
    fi

ADD main.py .

COPY source /app/source
COPY sql /app/sql

CMD ["python", "-u", "main.py"]