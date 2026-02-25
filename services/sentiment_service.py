# ===========================================
# Alpha Sentiment Engine — Sentiment Service
# ===========================================
# This is the "recipe book" — it knows HOW to score a headline.
#
# It's the same AI logic from your prototype.py, but now
# it lives in its own file so the worker can use it
# without knowing anything about APIs or scraping.
#
# HOW IT WORKS (same 4 steps as before):
#   1. Tokenize: headline text → numbers
#   2. Run the model: numbers → raw scores
#   3. Get probabilities: [positive, negative, neutral]
#   4. Calculate: score = positive - negative (-1 to +1)
# ===========================================

from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
import torch

from domain.models import NewsItem, SentimentResult


class SentimentService:
    """
    Loads the FinBERT model and scores headlines.

    Usage:
        service = SentimentService()             # loads model (slow, do once)
        result = service.analyze(news_item)      # scores headline (fast)
    """

    def __init__(self) -> None:
        """Load the FinBERT model. This is slow (~5 sec) — only do it once."""

        # YOUR custom-trained model (86.39% accuracy!)
        # Trained on 12,228 AI-verified financial texts
        model_name: str = "models/my_finbert"

        # Load the tokenizer (text → numbers translator)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the AI model (the brain)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Tell PyTorch we're scoring, not training
        self.model.eval()

    def analyze(self, news_item: NewsItem) -> SentimentResult:
        """
        Score a single headline.

        Args:
            news_item: A NewsItem with ticker and headline.

        Returns:
            A SentimentResult with the sentiment score (-1 to +1).
        """

        # Step 1: Turn headline → numbers
        inputs = self.tokenizer(
            news_item.headline,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Step 2: Run the AI (no_grad = save memory)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Step 3: Raw scores → probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)
        positive_prob: float = probabilities[0][0].item()
        negative_prob: float = probabilities[0][1].item()

        # Step 4: Single score from -1 to +1
        sentiment_score: float = round(positive_prob - negative_prob, 4)

        # Build and return the result
        return SentimentResult(
            ticker=news_item.ticker,
            sentiment_score=sentiment_score,
            headline=news_item.headline,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
