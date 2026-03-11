#!/bin/bash
# ==========================================
# Alpha Sentiment Engine — Container Startup
# Runs Flask dashboard + Live Scraper together
# ==========================================

echo "🚀 Starting Alpha Sentiment Engine..."

# Start the live scraper in the background
echo "📡 Starting V5 Live Scraper (background, every 30 min)..."
python live_scraper.py &
SCRAPER_PID=$!
echo "   Scraper PID: $SCRAPER_PID"

# Start the Flask dashboard in the foreground
echo "🌐 Starting Flask Dashboard on port 7860..."
python app.py

# If Flask exits, also kill the scraper
kill $SCRAPER_PID 2>/dev/null
