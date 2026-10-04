# TF 2.15 supports Python 3.9-3.11
FROM python:3.10-slim

WORKDIR /app

# install deps first so this layer is cached when only code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY configs/ configs/
COPY src/ src/
COPY models/production/ models/production/

ENV MODEL_DIR=models/production
EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]