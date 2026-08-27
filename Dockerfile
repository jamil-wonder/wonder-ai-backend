FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Install browser
RUN playwright install chromium

# Copy application
COPY . .
RUN chmod +x entrypoint.sh

# ROLE=web (default) runs gunicorn; ROLE=worker runs the background job
# process instead. See docker-compose.yml.
CMD ["./entrypoint.sh"]
