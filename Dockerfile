# Base image 
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Accept build-time argument
ARG DEVELOPMENT

# === Basic setup ===
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/code"
WORKDIR /code/watcher

# === Install PostgreSQL dev headers ===
RUN apt-get update && \
    apt-get install -y libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# === Install Microsoft Core Fonts (e.g., Arial, Times New Roman) ===
# Add key fonts including Times New Roman and Arial
ENV DEBIAN_FRONTEND=teletype
RUN apt update 
RUN yes | apt-get install msttcorefonts -y
ENV DEBIAN_FRONTEND=

# === Python requirements ===
COPY docker_requirements.txt /code/watcher/
RUN pip install --no-cache-dir -r docker_requirements.txt

# === Copy all files ===
COPY . /code/watcher/

# === Install Watcher == (editable if DEVELOPMENT=true)
RUN if [ "$DEVELOPMENT" = "true" ]; then \
        pip install -e .; \
    else \
        pip install .; \
    fi