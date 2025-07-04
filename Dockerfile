FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Install the package
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Run the dashboard
CMD ["streamlit", "run", "examples/dashboard_launcher.py", "--server.port=8501", "--server.address=0.0.0.0"]