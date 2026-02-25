FROM python:3.10-slim

# Install system dependencies (like SQLite)
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Set up user map to prevent Hugging Face permission errors
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

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
