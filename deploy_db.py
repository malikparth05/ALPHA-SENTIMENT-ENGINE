import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
REPO = "malikparth05/alpha-sentiment-engine"

print(f"🚀 Activating Hugging Face API Data Payload Sync...")

try:
    api = HfApi(token=TOKEN)
    for file in ["sentiment_data.json", "sentiment_headlines.json", "app.py"]:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=REPO,
                repo_type="space",
                commit_message=f"Cloud Sync: {file}"
            )
            print(f"✅ {file} uploaded.")
    print("✅ All data synced to cloud! Dashboard will restart.")
except Exception as e:
    print(f"❌ Upload Failed: {e}")
