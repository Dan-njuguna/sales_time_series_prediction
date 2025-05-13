FROM python:3.11-alpine AS builder

# setting up environment
RUN apt-get update && apt-get install -y build-essential \
    gcc \
    cmake

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Final image
FROM python:3.11-alpine

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

EXPOSE 8000

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]