FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y sqlite3 git && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container
COPY --chown=user . .

# Make startup script executable
RUN chmod +x start.sh

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860
ENV FLASK_APP=app.py

# Run both Flask dashboard + Live Scraper
CMD ["bash", "start.sh"]
