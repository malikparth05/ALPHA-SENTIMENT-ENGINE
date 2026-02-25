#!/usr/bin/env python3
# ===========================================
# Alpha Sentiment Engine — Live Scraper v3 (India Only)
# ===========================================
# FEATURES:
#   - Tracks 2,200+ Indian stocks (NSE)
#   - ⚡ Two-Level Scoring Strategy:
#       1. Company news (for the top Nifty 50)
#       2. Sector news (applied across all 2,200+ stocks)
#   - 💾 Saves to SQLite database
# ===========================================

import time
import json
import feedparser
import requests
import yfinance as yf
from urllib.parse import quote
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, Style, init

from services.sentiment_service import SentimentService
from services.database import save_score, save_average, get_stats
from domain.models import NewsItem

init(autoreset=True)


# ───────────────────────────────────────────
#  Settings
# ───────────────────────────────────────────
SCRAPE_INTERVAL: int = 1800  # 30 minutes 
HEADLINES_PER_SEARCH: int = 3
STOCKS_FILE = "data/indian_stocks.json"

# We only search specific company news for the top 50 to save time/API limits.
# The other 2,150+ stocks will get scored based on their SECTOR news.
TOP_COMPANIES = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"
]

# Map our internal sectors to good Google search queries
SECTOR_QUERIES = {
    "IT": "Indian IT sector stock market",
    "Banking": "Indian banking sector stock market RBI",
    "Pharma": "Indian pharma sector pharmaceutical stock",
    "Auto": "Indian auto sector automobile",
    "FMCG": "Indian FMCG consumer goods stock",
    "Energy": "Indian power energy oil gas sector stock",
    "Real Estate": "Indian real estate property sector stock",
    "Metal": "Indian metal steel mining sector stock",
    "Chemicals": "Indian chemical sector stock",
    "Logistics": "Indian logistics shipping port sector stock",
    "Telecom": "Indian telecom sector stock",
}

# ───────────────────────────────────────────
#  Load stock universe
# ───────────────────────────────────────────
def load_stocks():
    print("📥 Loading Indian stock database...")
    try:
        with open(STOCKS_FILE, "r") as f:
            data = json.load(f)
            return data["stocks"], data["sectors"]
    except Exception as e:
        print(f"❌ Could not load {STOCKS_FILE}. Did you run fetch_stocks.py? {e}")
        return [], {}

# ───────────────────────────────────────────
#  Source 1: Google News (FREE, UNLIMITED)
# ───────────────────────────────────────────
def fetch_google_news(query: str, count: int = 3) -> list[dict]:
    """Fetch headlines from Google News RSS."""
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}+news&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:count]:
            title = entry.get("title", "").split(" - ")[0] # Clean up source name
            source = entry.get("source", {}).get("title", "News") if hasattr(entry, "source") else "News"
            if title:
                results.append({"title": title, "source": f"📰 {source}"})
        return results
    except Exception:
        return []

# ───────────────────────────────────────────
#  Source 2: Reddit (FREE, UNLIMITED)
# ───────────────────────────────────────────
def fetch_reddit_posts(query: str, count: int = 2) -> list[dict]:
    """Fetch recent Reddit posts."""
    url = "https://www.reddit.com/search.json"
    params = {"q": f"{query} India market", "sort": "new", "limit": count, "t": "week"}
    headers = {"User-Agent": "AlphaSentimentEngine/1.0"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        results = []
        for post in data.get("data", {}).get("children", [])[:count]:
            title = post.get("data", {}).get("title", "")
            subreddit = post.get("data", {}).get("subreddit", "reddit")
            if title:
                results.append({"title": title, "source": f"💬 r/{subreddit}"})
        return results
    except Exception:
        return []

def fetch_all_headlines(query: str) -> list[dict]:
    """Fetch from Google News (reduced to 1 headline to save massive Cloud CPU time)."""
    return fetch_google_news(query, 1)

# ───────────────────────────────────────────
#  Scoring & helpers
# ───────────────────────────────────────────
def score_headline(service: SentimentService, headline: str) -> float:
    # We pass a dummy ticker since we are scoring the text itself
    item = NewsItem(ticker="DUMMY", headline=headline)
    result = service.analyze(item)
    return result.sentiment_score

def get_sentiment_color(score: float) -> str:
    if score > 0.3: return Fore.GREEN
    elif score < -0.3: return Fore.RED
    else: return Fore.YELLOW

def get_sentiment_label(score: float) -> str:
    if score > 0.5: return "🟢 V.Bullish"
    elif score > 0.3: return "🟢 Bullish"
    elif score > -0.3: return "🟡 Neutral"
    elif score > -0.5: return "🔴 Bearish"
    else: return "🔴 V.Bearish"

# ───────────────────────────────────────────
#  The Main Scrape Cycle
# ───────────────────────────────────────────
def run_scrape_cycle(service: SentimentService, stocks: list[dict], sectors_map: dict) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    start_time = time.time()

    print("\n" + "=" * 80)
    print(f"  📡 LIVE SCRAPE — {now}")
    print(f"  📊 Tracking {len(stocks)} Indian stocks")
    print(f"  🧠 Strategy: Two-Level Scoring (Sector + Company)")
    print("=" * 80)

    # Dictionary to hold the final score for EVERY single company
    final_company_scores = {}
    
    # Dictionary to hold the baseline sector scores
    sector_scores = {}

    # 1️⃣ LEVEL 1: SECTOR SENTIMENT 
    print("\n  🏢 LEVEL 1: Scanning Sector Sentiments...")
    for sector, count in sorted([(k, len(v)) for k, v in sectors_map.items()], key=lambda x: -x[1])[:10]:
        query = SECTOR_QUERIES.get(sector, f"Indian {sector} sector stock market")
        headlines = fetch_all_headlines(query)
        
        if not headlines:
            sector_scores[sector] = 0.0
            continue
            
        scores = []
        for article in headlines:
            score = score_headline(service, article["title"])
            scores.append(score)
            # Save sector news to DB under the sector name
            save_score(f"SECTOR_{sector}", article["title"], score, article["source"])
            
        avg_score = sum(scores) / len(scores)
        sector_scores[sector] = avg_score
        
        # Pretty print
        color = get_sentiment_color(avg_score)
        print(f"     {sector:<15} ({count} stocks): {color}{avg_score:+.4f}{Style.RESET_ALL}  {get_sentiment_label(avg_score)}")

    # Assign base sector scores to ALL 2,200+ companies
    for stock in stocks:
        symbol = stock["symbol"]
        sector = stock.get("sector", "General")
        # Give it the sector score, or neutral 0.0 if not found
        final_company_scores[symbol] = [sector_scores.get(sector, 0.0)]

    # 2️⃣ LEVEL 2: DIRECT COMPANY NEWS (Top Companies Only)
    print("\n  🏢 LEVEL 2: Scanning Direct Company News (Top 20)...")
    for symbol in TOP_COMPANIES:
        # Find the stock object to get full name
        stock_obj = next((s for s in stocks if s["symbol"] == symbol), None)
        if not stock_obj: continue
        
        company_name = stock_obj["name"]
        headlines = fetch_all_headlines(f"{company_name} stock")
        
        if not headlines:
            continue
            
        scores = []
        for article in headlines:
            score = score_headline(service, article["title"])
            scores.append(score)
            save_score(symbol, article["title"], score, article["source"])
            
        avg_score = sum(scores) / len(scores)
        color = get_sentiment_color(avg_score)
        
        # Add to the company's score list (combining with sector base)
        final_company_scores[symbol].append(avg_score)
        
        # Display the direct company news findings
        print(f"     {symbol:<15} Direct News: {color}{avg_score:+.4f}{Style.RESET_ALL} (from {len(headlines)} headlines)")

    # 3️⃣ FINALIZE AND SAVE ALL 2,200+ SCORES
    print("\n  💾 LEVEL 3: Saving all 2,200+ companies to Database...")
    saved_count = 0
    
    # Calculate final average (sector + direct combined) and save
    top_bullish = []
    top_bearish = []

    for symbol, scores in final_company_scores.items():
        if not scores: continue
        
        # Final score is the average of (sector base + any direct news)
        # If it only has sector news, it just takes that score.
        final_avg = sum(scores) / len(scores)
        
        # Don't save pure 0.0 (no data)
        if final_avg != 0.0:
            save_average(symbol, final_avg, len(scores))
            saved_count += 1
            
            # Keep track for summary
            if final_avg > 0.3: top_bullish.append((symbol, final_avg))
            if final_avg < -0.3: top_bearish.append((symbol, final_avg))

    # Calculate time taken
    elapsed = time.time() - start_time

    # 4️⃣ SUMMARY DASHBOARD
    print("\n" + "=" * 80)
    print("  📊 CYCLE COMPLETE: Two-Level Scoring Architecture")
    print("=" * 80)
    
    print(f"  ✅ Scored {saved_count} companies based on Sector + Company news.")
    
    print("\n  🟢 TOP BULLISH COMPANIES (Current Cycle):")
    for symbol, score in sorted(top_bullish, key=lambda x: -x[1])[:5]:
        stock_obj = next((s for s in stocks if s["symbol"] == symbol), None)
        name = stock_obj["name"] if stock_obj else ""
        print(f"     {symbol:<12} {Fore.GREEN}{score:+.4f}{Style.RESET_ALL}  {name[:30]}")
        
    print("\n  🔴 TOP BEARISH COMPANIES (Current Cycle):")
    for symbol, score in sorted(top_bearish, key=lambda x: x[1])[:5]:
        stock_obj = next((s for s in stocks if s["symbol"] == symbol), None)
        name = stock_obj["name"] if stock_obj else ""
        print(f"     {symbol:<12} {Fore.RED}{score:+.4f}{Style.RESET_ALL}  {name[:30]}")

    print("\n" + "=" * 80)
    stats = get_stats()
    print(f"  💾 Database Totals: {stats['total_scores']} headlines | {stats['total_averages']} daily averages recorded")
    print(f"  ⏱️  Cycle completed in {elapsed:.1f} seconds")
    print("=" * 80)


def main():
    print("=" * 80)
    print("  🚀 ALPHA SENTIMENT ENGINE — INDIA LIVE MODE v3")
    print("=" * 80)

    stocks, sectors = load_stocks()
    if not stocks:
        return

    print(f"  Tracking: {len(stocks)} Indian stocks across {len(sectors)} sectors")
    print(f"  Strategy: Two-Level Scoring (Broad Sector Base + Deep Company Focus)")
    print(f"  Interval: Every {SCRAPE_INTERVAL // 60} minutes")
    print(f"  AI Model: models/my_finbert (86.39% accuracy)")
    print("=" * 80)

    print("\n⏳ Loading YOUR custom-trained FinBERT model...")
    service = SentimentService()
    print("✅ AI loaded and ready!\n")

    run_scrape_cycle(service, stocks, sectors)

    print(f"\n💤 Sleeping for {SCRAPE_INTERVAL // 60} minutes...")
    print("   (Press Ctrl+C to stop)\n")

    try:
        while True:
            time.sleep(SCRAPE_INTERVAL)
            run_scrape_cycle(service, stocks, sectors)
            print(f"\n💤 Sleeping for {SCRAPE_INTERVAL // 60} minutes...")
            print("   (Press Ctrl+C to stop)\n")
    except KeyboardInterrupt:
        print("\n\n👋 Stopped. Goodbye!")


if __name__ == "__main__":
    main()
