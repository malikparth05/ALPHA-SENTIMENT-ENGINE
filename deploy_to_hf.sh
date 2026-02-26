#!/bin/bash
# ==========================================
# ONE-CLICK DEPLOY TO HUGGING FACE
# ==========================================
# This script securely pushes all your latest 
# files (and the sentiment_data.db database)
# directly to the Hugging Face cloud, bypassing
# any giant dataset history conflicts.

echo "🚀 Preparing Hugging Face Cloud Deployment..."

git checkout --orphan hf-auto-deploy
git add .
git commit -m "Cloud Data Sync"
git push -f hf hf-auto-deploy:main
git checkout main
git branch -D hf-auto-deploy

echo "✅ Deployment Successful! Check your HF Dashboard."
