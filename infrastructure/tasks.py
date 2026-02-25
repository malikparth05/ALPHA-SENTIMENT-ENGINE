# ===========================================
# Alpha Sentiment Engine — Worker Task
# ===========================================
# This is the "cook" — the function that actually runs
# when a headline arrives in the Redis queue.
#
# Flow:
#   1. A headline dict arrives from the queue
#   2. Convert it to a NewsItem object
#   3. Call the AI service to score it
#   4. Convert the result back to a dict
#   5. Send it back
#
# IMPORTANT: The AI model is loaded ONCE when the worker starts.
# Every headline after that reuses the same model (fast!).
# ===========================================

from typing import Optional

from infrastructure.celery_app import app
from domain.models import NewsItem, SentimentResult
from services.sentiment_service import SentimentService

# ---- Lazy Singleton ----
# The AI model starts as None. The FIRST task that runs
# creates the SentimentService (loads the model).
# Every task after that reuses it. This way we don't load
# the model every time (that would be painfully slow).
_service: Optional[SentimentService] = None


def _get_service() -> SentimentService:
    """Get (or create) the AI service. Loads model on first call only."""
    global _service
    if _service is None:
        print("🔄 Loading FinBERT model (first time only, please wait)...")
        _service = SentimentService()
        print("✅ FinBERT model loaded and ready!")
    return _service


@app.task(name="analyze_sentiment")
def analyze_sentiment_task(news_dict: dict) -> dict:
    """
    Celery task: score a headline's sentiment.

    This function is called AUTOMATICALLY by Celery when
    a new headline appears in the Redis queue.

    Args:
        news_dict: {"ticker": "AAPL", "headline": "Apple beats..."}

    Returns:
        {"ticker": "AAPL", "sentiment_score": 0.89,
         "headline": "Apple beats...", "timestamp": "2026-..."}
    """

    # Step 1: Convert raw dict → NewsItem object
    news_item: NewsItem = NewsItem.from_dict(news_dict)

    # Step 2: Get the AI service (loads model if first time)
    service: SentimentService = _get_service()

    # Step 3: Score the headline
    result: SentimentResult = service.analyze(news_item)

    # Step 4: Convert result back to dict (Celery needs JSON-friendly data)
    return result.to_dict()
