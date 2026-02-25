# ===========================================
# Alpha Sentiment Engine — Data Models
# ===========================================
# These define the SHAPE of our data.
# Think of them as templates / forms:
#   - NewsItem:        the ORDER (what goes into the queue)
#   - SentimentResult: the FINISHED DISH (what comes out)
#
# Both sides (scraper + AI worker) use these same templates
# so they agree on what the data looks like. That's the
# "contract" between them.
# ===========================================

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class NewsItem:
    """
    The INPUT — a headline we want to score.

    Example:
        item = NewsItem(ticker="AAPL", headline="Apple beats earnings")
    """
    ticker: str      # stock symbol, e.g. "AAPL"
    headline: str    # the news headline text

    def to_dict(self) -> dict:
        """Convert to a plain dictionary so we can send it as JSON."""
        return {
            "ticker": self.ticker,
            "headline": self.headline,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NewsItem":
        """Create a NewsItem from a plain dictionary."""
        return cls(
            ticker=data["ticker"],
            headline=data["headline"],
        )


@dataclass
class SentimentResult:
    """
    The OUTPUT — a scored headline.

    Example:
        result = SentimentResult(
            ticker="AAPL",
            sentiment_score=0.89,
            headline="Apple beats earnings",
            timestamp="2026-02-22T15:00:00+00:00"
        )
    """
    ticker: str              # stock symbol
    sentiment_score: float   # -1.0 (very bad) to +1.0 (very good)
    headline: str            # the original headline
    timestamp: str           # when the analysis happened (ISO 8601)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary (JSON-ready)."""
        return {
            "ticker": self.ticker,
            "sentiment_score": self.sentiment_score,
            "headline": self.headline,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SentimentResult":
        """Create a SentimentResult from a plain dictionary."""
        return cls(
            ticker=data["ticker"],
            sentiment_score=data["sentiment_score"],
            headline=data["headline"],
            timestamp=data["timestamp"],
        )
