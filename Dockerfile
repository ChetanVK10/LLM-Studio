# Use Python 3.10 slim base image for runtime optimization
FROM python:3.10-slim

# Prevent Python from writing .pyc bytecodes and configure unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

WORKDIR /app

# Install compilation helpers and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirement lists
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository code files
COPY . /app/

# Expose standard Streamlit port
EXPOSE 8501

# Run system validation checks on container startup prior to launching Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
