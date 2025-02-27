FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY setup.py .
COPY requirements.txt .
RUN pip install -e .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Expose ports
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
ENV QUANTIZE="True"
ENV PORT=5000

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
if [[ "$1" == "api" ]]; then\n\
  exec gunicorn --bind 0.0.0.0:${PORT:-5000} quick_wins.marketing_chatbot.api:app\n\
elif [[ "$1" == "dashboard" ]]; then\n\
  exec streamlit run quick_wins/dashboard/app.py\n\
else\n\
  echo "Available commands: api, dashboard"\n\
  exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"] 