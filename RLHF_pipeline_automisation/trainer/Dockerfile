# # builder
# FROM python:3.10-slim-bullseye AS builder
# WORKDIR /app
# COPY requirements-trainer.txt .
# RUN apt-get update \
#     && apt-get install -y git \
#     && pip install --no-cache-dir -r requirements-trainer.txt

# # final
# FROM python:3.10-slim-bullseye
# WORKDIR /app
# COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
# COPY train_dpo.py .

# ENV PORT=8080
# ENTRYPOINT ["python", "train_dpo.py"]


# builder stage
FROM python:3.10-slim-bullseye AS builder
WORKDIR /app

RUN apt-get update \
 && apt-get install -y git build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-trainer.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r requirements-trainer.txt

# final stage
FROM python:3.10-slim-bullseye
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY train_dpo.py .

ENV PORT=8080
ENTRYPOINT ["python", "train_dpo.py"]
