#!/usr/bin/env python3
# ===========================================
#  Alpha Sentiment Engine — PROTOTYPE (Day 2)
# ===========================================
#  GOAL: Pull stock price + headlines, then
#        score each headline with FinBERT AI.
#
#  Before running, install dependencies:
#    pip install requests pandas transformers torch
#
#  Then just run:
#    python prototype.py
# ===========================================

import requests
import pandas as pd
from datetime import datetime

# These two lines load the FinBERT AI tools:
# - AutoTokenizer:  converts text → numbers the AI can read
# - AutoModel...:   the actual AI brain that does the scoring
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
import torch

# ───────────────────────────────────────────
#  🔑  PASTE YOUR API KEYS HERE
# ───────────────────────────────────────────
#  1. Alpha Vantage (free): https://www.alphavantage.co/support/#api-key
#  2. NewsAPI       (free): https://newsapi.org/register
# ───────────────────────────────────────────
ALPHA_VANTAGE_KEY: str = "4UNJIJF65URP0KDY"
NEWS_API_KEY: str = "5f08b14eae02463f86d53dfd190fe74f"

# The stock ticker you want to look up
TICKER: str = "AAPL"

# ───────────────────────────────────────────
#  Ticker → Company Name mapping
# ───────────────────────────────────────────
#  NewsAPI can't search "AAPL" — no article says that.
#  We need the real company name so it finds actual headlines.
#  Add more tickers here as you need them!
# ───────────────────────────────────────────
COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "META": "Meta",
}


# ──────────────────────────────────────────────────────────
#  PART 1 — Get the current stock price from Alpha Vantage
# ──────────────────────────────────────────────────────────
def get_stock_price(ticker: str) -> dict:
    """
    Fetch the latest stock price for a given ticker symbol.

    Returns a dict like:
      {"price": 189.45, "volume": 52341234, "updated": "2026-02-21 12:30:00"}
    """

    url: str = "https://www.alphavantage.co/query"
    params: dict = {
        "function": "GLOBAL_QUOTE",   # gives us the latest price
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_KEY,
    }

    print(f"📡 Fetching stock price for {ticker}...")
    response = requests.get(url, params=params)
    data: dict = response.json()

    # Alpha Vantage wraps everything under "Global Quote"
    quote: dict = data.get("Global Quote", {})

    if not quote:
        print(f"   ⚠️  No data returned. Check your API key and ticker.")
        return {"price": 0.0, "volume": 0, "updated": "N/A"}

    return {
        "price": float(quote.get("05. price", 0)),
        "volume": int(quote.get("06. volume", 0)),
        "updated": quote.get("07. latest trading day", "N/A"),
    }


# ──────────────────────────────────────────────────────────
#  PART 2 — Get the top 5 news headlines from NewsAPI
# ──────────────────────────────────────────────────────────
def get_news_headlines(query: str, count: int = 5) -> list[dict]:
    """
    Fetch the latest news headlines for a search query.

    Returns a list of dicts like:
      [{"title": "Apple beats...", "source": "Reuters", "url": "https://..."}]
    """

    url: str = "https://newsapi.org/v2/everything"
    params: dict = {
        "q": query,
        "sortBy": "publishedAt",   # newest first
        "pageSize": count,
        "language": "en",
        "apiKey": NEWS_API_KEY,
    }

    print(f"📰 Fetching top {count} headlines for '{query}'...")
    response = requests.get(url, params=params)
    data: dict = response.json()

    articles: list = data.get("articles", [])

    if not articles:
        print(f"   ⚠️  No articles found. Check your API key.")
        return []

    # Pull out just what we need from each article
    headlines: list[dict] = []
    for article in articles:
        headlines.append({
            "title": article.get("title", "No title"),
            "source": article.get("source", {}).get("name", "Unknown"),
            "published": article.get("publishedAt", "N/A"),
            "url": article.get("url", ""),
        })

    return headlines


# ──────────────────────────────────────────────────────────
#  PART 2.5 — Score a headline with FinBERT AI
# ──────────────────────────────────────────────────────────
#  This is the NEW part! We load the AI model once,
#  then use it to score every headline.
# ──────────────────────────────────────────────────────────

# Load the model ONCE when the script starts (not inside the function).
# This takes a few seconds the first time (downloads ~440 MB).
# After that, it's cached on your Mac.
print("🤖 Loading FinBERT AI model (first time may take a minute)...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()   # tell PyTorch "we're just scoring, not training"
print("✅ FinBERT loaded and ready!\n")


def score_sentiment(headline: str) -> float:
    """
    Score a single headline using FinBERT.

    How it works (4 micro-steps):
      1. Tokenize: turn the headline text into numbers
      2. Run the model: feed numbers into the AI
      3. Get probabilities: [positive, negative, neutral]
      4. Calculate score: positive - negative = score from -1 to +1

    Args:
        headline: the news headline text

    Returns:
        A float from -1.0 (very negative) to +1.0 (very positive)
    """

    # Step 1: Turn headline text → numbers (tokens)
    inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Step 2: Run the AI model (torch.no_grad = save memory, we're not training)
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 3: Convert raw scores → probabilities (they'll add up to 1.0)
    #   outputs.logits shape: [1, 3] → [positive, negative, neutral]
    probs = torch.softmax(outputs.logits, dim=1)
    positive: float = probs[0][0].item()
    negative: float = probs[0][1].item()
    # neutral = probs[0][2].item()  ← we don't need this

    # Step 4: Single score = positive minus negative
    score: float = round(positive - negative, 4)
    return score


def sentiment_emoji(score: float) -> str:
    """Return a colored emoji based on the sentiment score."""
    if score > 0.15:
        return "🟢"
    elif score < -0.15:
        return "🔴"
    else:
        return "🟡"


# ──────────────────────────────────────────────────────────
#  PART 3 — Print everything cleanly
# ──────────────────────────────────────────────────────────
def main() -> None:
    """The main function — ties everything together."""

    print()
    print("=" * 60)
    print("  🚀  Alpha Sentiment Engine — Prototype")
    print(f"  📅  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # ---- Stock Price ----
    price_data: dict = get_stock_price(TICKER)

    print()
    print(f"  💰  {TICKER} Stock Price")
    print(f"  ─────────────────────────")
    print(f"  Price:      ${price_data['price']:.2f}")
    print(f"  Volume:     {price_data['volume']:,}")
    print(f"  Last Trade: {price_data['updated']}")
    print()

    # ---- News Headlines ----
    # Use the company name (not ticker) for searching — "Apple stock" works,
    # "AAPL" does not, because articles use the company name.
    company: str = COMPANY_NAMES.get(TICKER, TICKER)
    search_term: str = f"{company} stock"
    headlines: list[dict] = get_news_headlines(search_term)

    print(f"  📰  Top {len(headlines)} Headlines for {TICKER} (with AI Sentiment)")
    print(f"  ─────────────────────────")
    for i, article in enumerate(headlines, start=1):
        # ✨ NEW: Score each headline with FinBERT!
        score: float = score_sentiment(article["title"])
        emoji: str = sentiment_emoji(score)

        print(f"  {i}. {article['title']}")
        print(f"     Source: {article['source']}  |  {article['published'][:10]}")
        print(f"     {emoji} Sentiment: {score:+.4f}")
        print()

        # Save the score in the article dict (for the table later)
        article["sentiment"] = score

    # ---- Show as a pandas DataFrame (bonus!) ----
    if headlines:
        print("  📊  Headlines + Sentiment Table")
        print(f"  ─────────────────────────")
        df = pd.DataFrame(headlines)
        print(df[["title", "source", "sentiment"]].to_string(index=False))
        print()

    print("=" * 60)
    print("  ✅  Prototype complete! Price + News + AI Sentiment")
    print("  👉  Next step: build the async pipeline with Redis + Celery.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
