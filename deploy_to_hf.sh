#!/bin/bash
# ==========================================
# Deploy Alpha Sentiment Engine to Hugging Face Spaces
# This pushes ALL files (code + model) for 24/7 cloud operation
# ==========================================

set -e

REPO_ID="malikparth05/alpha-sentiment-engine"
SPACE_DIR="/tmp/hf_space_deploy"

echo "🚀 DEPLOYING ALPHA SENTIMENT ENGINE TO HUGGING FACE SPACES"
echo "   Repo: $REPO_ID"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not set. Run: export HF_TOKEN=your_token"
    exit 1
fi

# Clean previous deploy dir
rm -rf "$SPACE_DIR"
mkdir -p "$SPACE_DIR"

# Clone the HF Space repo
echo "📥 Cloning HF Space repository..."
cd "$SPACE_DIR"
git clone "https://malikparth05:${HF_TOKEN}@huggingface.co/spaces/${REPO_ID}" repo
cd repo

# Install Git LFS for large model files
echo "📦 Setting up Git LFS for model files..."
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.arrow"
git add .gitattributes

# Copy project files
PROJECT_DIR="/Users/parthmalik/Documents/PORTFOLIO"
echo "📋 Copying project files..."

# Core Python files
cp "$PROJECT_DIR/app.py" .
cp "$PROJECT_DIR/live_scraper.py" .
cp "$PROJECT_DIR/start.sh" .
cp "$PROJECT_DIR/requirements.txt" .
cp "$PROJECT_DIR/Dockerfile" .

# Data files
mkdir -p data
cp "$PROJECT_DIR/data/indian_stocks.json" data/
cp "$PROJECT_DIR/data/fetch_stocks.py" data/

# Services
mkdir -p services
cp "$PROJECT_DIR/services/database.py" services/
cp "$PROJECT_DIR/services/__init__.py" services/ 2>/dev/null || touch services/__init__.py

# Static files
mkdir -p static
cp "$PROJECT_DIR/static/style.css" static/
cp "$PROJECT_DIR/static/app.js" static/

# Templates
mkdir -p templates
cp "$PROJECT_DIR/templates/index.html" templates/

# Model files (large - uses Git LFS)
echo "🧠 Copying FinBERT model (2.9GB via Git LFS)..."
mkdir -p models/my_finbert
cp "$PROJECT_DIR/models/my_finbert/config.json" models/my_finbert/
cp "$PROJECT_DIR/models/my_finbert/model.safetensors" models/my_finbert/
cp "$PROJECT_DIR/models/my_finbert/tokenizer.json" models/my_finbert/
cp "$PROJECT_DIR/models/my_finbert/tokenizer_config.json" models/my_finbert/
cp "$PROJECT_DIR/models/my_finbert/training_args.bin" models/my_finbert/

# Export current DB as seed data (optional)
if [ -f "$PROJECT_DIR/sentiment_data.db" ]; then
    echo "💾 Copying seed database..."
    cp "$PROJECT_DIR/sentiment_data.db" .
fi

# Create README for HF Space
cat > README.md << 'EOF'
---
title: Alpha Sentiment Engine
emoji: ⚡
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: true
---

# Alpha Sentiment Engine

AI-powered Indian stock market sentiment analysis using a custom-trained FinBERT model (86.39% accuracy).

- 🧠 Custom FinBERT model fine-tuned on financial news
- 📡 Live scraping from Google News + Yahoo Finance RSS
- 📊 V5 Three-Phase Hybrid Architecture (sector + direct + hybrid scoring)
- 🎯 561+ Indian companies scored with confidence levels
- 📈 Real-time stock price correlation
EOF

echo ""
echo "📤 Pushing to Hugging Face Spaces..."
git add -A
git status
git commit -m "V5 Deploy: Three-Phase Hybrid Scraper + 24/7 Live Operation"
git push

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "   🌐 Your Space: https://huggingface.co/spaces/${REPO_ID}"
echo "   ⏱️  Building... check back in 5-10 minutes"
echo "   📡 Scraper will auto-run every 30 minutes inside the container"
