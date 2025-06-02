FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

ENV CUDA_VISIBLE_DEVICES=""

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \	
    libgl1-mesa-glx \
    libglib2.0-0 \
    zlib1g-dev \
    gcc \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --upgrade pip
    
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --upgrade --no-deps simple-lama-inpainting
# simple-lama-inpainting 0.1.2 is fully compatible with pillow 11.4.1 but it throws a dependency error for pillow <10.0.0
# just ignore it!
# running tests locally before i open a pr with the repo

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "challenge.py"]
