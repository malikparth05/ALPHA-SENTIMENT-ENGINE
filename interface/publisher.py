# ===========================================
# Alpha Sentiment Engine — Publisher (The Waiter)
# ===========================================
# This is what your SCRAPER uses to send headlines
# into the pipeline. Just one function:
#
#     publish_news("AAPL", "Apple beats earnings")
#
# That's it! The scraper doesn't need to know about
# Redis, Celery, or FinBERT. It just calls this function
# and the headline gets queued for scoring.
# ===========================================

from celery.result import AsyncResult

from domain.models import NewsItem
from infrastructure.tasks import analyze_sentiment_task


def publish_news(ticker: str, headline: str) -> AsyncResult:
    """
    Send a headline into the sentiment analysis pipeline.

    This is THE function your scraper calls.

    Args:
        ticker:   Stock symbol, e.g. "AAPL"
        headline: The news headline text

    Returns:
        An AsyncResult — a "receipt" you can use to check
        if the analysis is done and get the result.

    Example:
        result = publish_news("AAPL", "Apple beats earnings")
        sentiment = result.get(timeout=30)  # wait up to 30 sec
        print(sentiment)
    """

    # Create a NewsItem and convert to dict
    news_item: NewsItem = NewsItem(ticker=ticker, headline=headline)

    # Send it to the Celery queue (non-blocking — returns immediately)
    # .delay() is shorthand for .apply_async()
    async_result: AsyncResult = analyze_sentiment_task.delay(news_item.to_dict())

    return async_result
