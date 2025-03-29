FROM python:3.10-slim


WORKDIR /app

# Copy requirements first for better caching
COPY requirenments.txt .
RUN pip install --no-cache-dir -r requirenments.txt

# Copy application code
COPY app.py .
COPY service-account-key.json .

# Set the environment variable for GCP authentication
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json

# Cloud Run will set PORT environment variable, default to 8080
ENV PORT=8080

# Run the application
CMD exec python app.py