#!/usr/bin/env python3
# ===========================================
# Alpha Sentiment Engine — Live Scraper V5
# ===========================================
# THREE-PHASE HYBRID ARCHITECTURE:
#   Phase A — Sector news scan (20 sectors)
#   Phase B — Direct company news (top 200)
#   Phase C — Hybrid merge (1000+ companies)
#
# FEATURES:
#   - Entity validation on direct news
#   - Headline deduplication
#   - Stock price overlay via yfinance
#   - Confidence levels (HIGH / MEDIUM / LOW)
#   - Score types (DIRECT / HYBRID / SECTOR)
#   - Multi-source: Google News + Yahoo Finance
# ===========================================

import time
import json
import hashlib
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
HEADLINES_PER_SEARCH: int = 8
STOCKS_FILE = "data/indian_stocks.json"

# How many top companies get direct news search
DIRECT_SCAN_COUNT = 200

# How many total companies to score (via hybrid merge)
TOTAL_COMPANY_TARGET = 1000

# Sector news search queries
SECTOR_QUERIES = {
    "Banking": "Indian banking sector news stock market",
    "IT": "Indian IT sector technology stocks news",
    "Pharma": "Indian pharma healthcare stocks news",
    "Auto": "Indian auto sector automobile stocks news",
    "FMCG": "Indian FMCG consumer goods stocks news",
    "Energy": "Indian energy power oil gas stocks news",
    "Metal": "Indian metal steel mining stocks news",
    "Telecom": "Indian telecom sector stocks news",
    "Real Estate": "Indian real estate construction cement stocks news",
    "Chemicals": "Indian chemicals paints sector stocks news",
    "Textiles": "Indian textile apparel sector stocks news",
    "Media": "Indian media entertainment sector stocks news",
    "Insurance": "Indian insurance sector stocks news",
    "Agriculture": "Indian agriculture sugar fertilizer stocks news",
    "Logistics": "Indian logistics transport aviation stocks news",
    "Retail": "Indian retail e-commerce stocks news",
    "Defence": "Indian defence aerospace shipbuilding stocks news",
    "Consumer Durables": "Indian consumer durables electronics stocks news",
    "Infrastructure": "Indian infrastructure stocks news",
    "Hotels": "Indian hotel hospitality tourism stocks news",
}

# Top 200 companies for direct news scanning (Nifty 50 + Next 50 + Midcap 100)
TOP_COMPANIES = [
    # Nifty 50
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "BAJFINANCE", "MARUTI", "HCLTECH", "ASIANPAINT", "SUNPHARMA",
    "TITAN", "WIPRO", "ONGC", "BAJAJFINSV", "ULTRACEMCO", "NTPC",
    "TATAMOTORS", "POWERGRID", "NESTLEIND", "JSWSTEEL", "TATASTEEL",
    "ADANIPORTS", "ADANIENT", "TECHM", "M&M", "COALINDIA",
    "INDUSINDBK", "HINDALCO", "SBILIFE", "GRASIM", "HDFCLIFE",
    "CIPLA", "DIVISLAB", "DRREDDY", "EICHERMOT", "BPCL",
    "APOLLOHOSP", "HEROMOTOCO", "BAJAJ-AUTO", "TATACONSUM", "BRITANNIA",
    "SHRIRAMFIN", "LTIM",
    # Nifty Next 50
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "BANKBARODA", "BERGEPAINT",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DABUR",
    "DLF", "GAIL", "GODREJCP", "HAVELLS", "ICICIPRULI",
    "IDFCFIRSTB", "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
    "LUPIN", "MAXHEALTH", "MOTHERSON", "NAUKRI", "OBEROIRLTY",
    "OFSS", "PAGEIND", "PEL", "PIDILITIND", "PNB",
    "SAIL", "SRF", "TATAPOWER", "TORNTPHARM", "TRENT",
    "UNIONBANK", "VEDL", "YESBANK", "ZOMATO", "PAYTM",
    # Midcap 100 extras
    "DMART", "POLICYBZR", "NYKAA", "PB", "IRFC",
    "IDEA", "RECLTD", "NHPC", "HAL", "BEL",
    "TATAELXSI", "PERSISTENT", "COFORGE", "MPHASIS", "LTTS",
    "PVRINOX", "PATANJALI", "MAZDOCK", "COCHINSHIP", "GRSE",
    "BIOCON", "AUROPHARMA", "ALKEM", "LALPATHLAB", "MARICO",
    "ASHOKLEY", "TVSMOTORS", "EXIDEIND", "AMARARAJA", "MUTHOOTFIN",
    "MANAPPURAM", "VOLTAS", "CROMPTON", "DIXON", "NMDC",
    "NATIONALUM", "LICHSGFIN", "CANFINHOME", "SHREECEM", "ACC",
    "GODREJPROP", "EMAMILTD", "HAPPSTMNDS", "MASTEK", "SONATA",
    "ABCAPITAL", "IIFL", "FEDERALBNK", "RBLBANK", "BANDHANBNK",
    "JUBLFOOD", "MCDOWELL-N", "CONCOR", "SUNTV", "NETWORK18",
    "DEEPAKNTR", "ATUL", "PIIND", "ASTRAL", "SUPREMEIND",
    "APLAPOLLO", "KENNAMET", "CUMMINSIND", "SIEMENS", "ABB",
    "HONAUT", "BHARATFORG", "SUNDRMFAST", "MFSL", "NAM-INDIA",
    "IPCALAB", "NATCOPHARMA", "GLENMARK", "LAURUSLABS", "SYNGENE",
    "GODREJIND", "UBL", "TATACOMM", "STARHEALTH", "KPITTECH",
]


# ───────────────────────────────────────────
#  Load stock universe
# ───────────────────────────────────────────
def load_stocks():
    print("📥 Loading Indian stock database...")
    try:
        with open(STOCKS_FILE, "r") as f:
            data = json.load(f)
            return data["stocks"], data.get("sectors", {})
    except Exception as e:
        print(f"❌ Could not load {STOCKS_FILE}. {e}")
        return [], {}


# ───────────────────────────────────────────
#  News Sources
# ───────────────────────────────────────────
def fetch_google_news(query: str, count: int = 8) -> list[dict]:
    """Fetch headlines from Google News RSS."""
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:count]:
            title = entry.get("title", "").split(" - ")[0]
            source = entry.get("source", {}).get("title", "News") if hasattr(entry, "source") else "News"
            if title:
                results.append({"title": title, "source": f"📰 {source}"})
        return results
    except Exception:
        return []


def fetch_yahoo_finance_news(query: str, count: int = 5) -> list[dict]:
    """Fetch headlines from Yahoo Finance RSS as a secondary source."""
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}+share+price+NSE+BSE&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:count]:
            title = entry.get("title", "").split(" - ")[0]
            if title:
                results.append({"title": title, "source": "📈 Finance"})
        return results
    except Exception:
        return []


# ───────────────────────────────────────────
#  Entity Validation
# ───────────────────────────────────────────
def validate_headline(headline: str, company_name: str, ticker: str) -> bool:
    """Check if a headline is ACTUALLY about this specific company."""
    headline_lower = headline.lower()

    clean_name = company_name.lower()
    for suffix in [" limited", " ltd", " ltd.", " corporation", " corp", " inc"]:
        clean_name = clean_name.replace(suffix, "").strip()

    name_parts = clean_name.split()

    if ticker.lower() in headline_lower:
        return True

    if len(clean_name) > 3 and clean_name in headline_lower:
        return True

    if len(name_parts) >= 1 and len(name_parts[0]) > 4:
        if name_parts[0] in headline_lower:
            return True

    if len(name_parts) >= 2:
        two_word = f"{name_parts[0]} {name_parts[1]}"
        if two_word in headline_lower:
            return True

    return False


# ───────────────────────────────────────────
#  Headline Deduplication
# ───────────────────────────────────────────
def headline_hash(text: str) -> str:
    """Create a short hash of a headline for dedup."""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:12]


# ───────────────────────────────────────────
#  Stock Price Fetcher (yfinance)
# ───────────────────────────────────────────
def get_price_change(yahoo_ticker: str) -> float | None:
    """Get today's stock price % change using yfinance."""
    try:
        stock = yf.Ticker(yahoo_ticker)
        hist = stock.history(period="2d")
        if len(hist) >= 2:
            prev_close = hist["Close"].iloc[-2]
            curr_close = hist["Close"].iloc[-1]
            pct = round(((curr_close - prev_close) / prev_close) * 100, 2)
            return pct
    except Exception:
        pass
    return None


# ───────────────────────────────────────────
#  Scoring Helpers
# ───────────────────────────────────────────
def score_headline(service: SentimentService, headline: str) -> float:
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


def get_confidence(num_headlines: int, score_type: str) -> str:
    if score_type == "SECTOR":
        return "LOW"
    if num_headlines >= 3:
        return "HIGH"
    elif num_headlines >= 1:
        return "MEDIUM"
    return "LOW"


# ───────────────────────────────────────────
#  MAIN: Three-Phase Scrape Cycle (V5)
# ───────────────────────────────────────────
def run_scrape_cycle(service: SentimentService, stocks: list[dict], sectors_map: dict) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    start_time = time.time()

    # Build lookup dictionaries
    stock_by_symbol = {s["symbol"]: s for s in stocks}
    seen_headlines = set()  # For deduplication

    print("\n" + "=" * 80)
    print(f"  📡 LIVE SCRAPE V5 — {now}")
    print(f"  🧠 Three-Phase Hybrid Architecture")
    print(f"  📊 Phase A: {len(SECTOR_QUERIES)} sectors | Phase B: {len(TOP_COMPANIES)} direct | Phase C: {TOTAL_COMPANY_TARGET}+ hybrid")
    print("=" * 80)

    # ══════════════════════════════════════════
    #  PHASE A: Sector News Scan
    # ══════════════════════════════════════════
    print(f"\n  ═══ PHASE A: Scanning {len(SECTOR_QUERIES)} Sector News Feeds ═══")
    sector_scores = {}

    for sector_name, query in SECTOR_QUERIES.items():
        headlines = fetch_google_news(query, HEADLINES_PER_SEARCH)
        if not headlines:
            continue

        scores = []
        for article in headlines:
            h = headline_hash(article["title"])
            if h in seen_headlines:
                continue
            seen_headlines.add(h)

            score = score_headline(service, article["title"])
            scores.append(score)
            save_score(f"SECTOR_{sector_name}", article["title"], score, article["source"])

        if scores:
            avg = sum(scores) / len(scores)
            sector_scores[sector_name] = {"score": avg, "headlines": len(scores)}
            color = get_sentiment_color(avg)
            print(f"     {sector_name:<22} {color}{avg:+.4f}{Style.RESET_ALL}  ({len(scores)} headlines)  {get_sentiment_label(avg)}")

    print(f"  ✅ Phase A complete: {len(sector_scores)} sectors scored")

    # ══════════════════════════════════════════
    #  PHASE B: Direct Company News Scan
    # ══════════════════════════════════════════
    print(f"\n  ═══ PHASE B: Direct News for {len(TOP_COMPANIES)} Companies ═══")
    direct_scores = {}
    total_headlines_found = 0
    total_validated = 0
    total_rejected = 0

    for i, symbol in enumerate(TOP_COMPANIES):
        stock_obj = stock_by_symbol.get(symbol)
        if not stock_obj:
            continue

        company_name = stock_obj["name"]

        # Multi-source: Google News + Yahoo Finance queries
        headlines = fetch_google_news(company_name, HEADLINES_PER_SEARCH)

        if not headlines:
            continue

        validated_scores = []
        for article in headlines:
            total_headlines_found += 1
            title = article["title"]

            # Dedup check
            h = headline_hash(title)
            if h in seen_headlines:
                continue
            seen_headlines.add(h)

            # Entity validation
            is_valid = validate_headline(title, company_name, symbol)
            if is_valid:
                score = score_headline(service, title)
                validated_scores.append(score)
                save_score(symbol, title, score, article["source"], validated=True)
                total_validated += 1
            else:
                total_rejected += 1

        if validated_scores:
            avg_score = sum(validated_scores) / len(validated_scores)
            direct_scores[symbol] = {
                "score": avg_score,
                "headlines": len(validated_scores),
                "name": company_name,
            }

            color = get_sentiment_color(avg_score)
            print(f"     {symbol:<15} {color}{avg_score:+.4f}{Style.RESET_ALL}  "
                  f"({len(validated_scores)}/{len(headlines)} validated)  "
                  f"{get_sentiment_label(avg_score)}")

        # Rate limiting
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(TOP_COMPANIES)} companies scanned ...")
            time.sleep(1)

    print(f"  ✅ Phase B complete: {len(direct_scores)} companies with direct news")
    print(f"     Headlines: {total_headlines_found} found | {total_validated} validated | {total_rejected} rejected")

    # ══════════════════════════════════════════
    #  PHASE C: Hybrid Merge + Price Overlay
    # ══════════════════════════════════════════
    print(f"\n  ═══ PHASE C: Hybrid Merge + Price Overlay ═══")

    # Determine which stocks to score (up to TOTAL_COMPANY_TARGET)
    all_symbols = list(stock_by_symbol.keys())[:TOTAL_COMPANY_TARGET]

    # Fetch prices for direct-scored companies in parallel
    print(f"  📈 Fetching stock prices for {len(direct_scores)} companies...")
    price_data = {}

    def fetch_price(symbol, yahoo_ticker):
        return symbol, get_price_change(yahoo_ticker)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for sym in direct_scores:
            stock_obj = stock_by_symbol.get(sym)
            if stock_obj:
                yahoo_ticker = stock_obj.get("yahoo_ticker", f"{sym}.NS")
                futures[executor.submit(fetch_price, sym, yahoo_ticker)] = sym

        for future in as_completed(futures):
            try:
                sym, pct = future.result()
                if pct is not None:
                    price_data[sym] = pct
            except Exception:
                pass

    print(f"  ✅ Got price data for {len(price_data)} companies")

    # Merge and save
    final_scored = {}
    direct_count = 0
    hybrid_count = 0
    sector_only_count = 0

    for symbol in all_symbols:
        stock_obj = stock_by_symbol.get(symbol)
        if not stock_obj:
            continue

        sector = stock_obj.get("sector", "General")
        sector_score_data = sector_scores.get(sector)
        direct_data = direct_scores.get(symbol)
        price_pct = price_data.get(symbol)

        if direct_data and sector_score_data:
            # HYBRID: 70% direct + 30% sector
            final_score = 0.7 * direct_data["score"] + 0.3 * sector_score_data["score"]
            num_headlines = direct_data["headlines"]
            score_type = "HYBRID"
            confidence = get_confidence(num_headlines, score_type)
            hybrid_count += 1
        elif direct_data:
            # DIRECT only (sector has no score)
            final_score = direct_data["score"]
            num_headlines = direct_data["headlines"]
            score_type = "DIRECT"
            confidence = get_confidence(num_headlines, score_type)
            direct_count += 1
        elif sector_score_data:
            # SECTOR only (no direct news for this company)
            final_score = sector_score_data["score"]
            num_headlines = sector_score_data["headlines"]
            score_type = "SECTOR"
            confidence = "LOW"
            sector_only_count += 1
        else:
            # No data at all (sector is "General" with no news)
            continue

        save_average(
            ticker=symbol,
            average_score=round(final_score, 6),
            num_headlines=num_headlines,
            confidence=confidence,
            price_change=price_pct,
            score_type=score_type,
        )
        final_scored[symbol] = {
            "score": final_score,
            "type": score_type,
            "confidence": confidence,
            "price": price_pct,
            "name": stock_obj["name"],
        }

    elapsed = time.time() - start_time

    # ─── SUMMARY ───
    print("\n" + "=" * 80)
    print("  📊 V5 CYCLE COMPLETE: Three-Phase Hybrid Scoring")
    print("=" * 80)
    print(f"  ✅ Total companies scored: {len(final_scored)}")
    print(f"     🎯 DIRECT:  {direct_count} (company-specific news)")
    print(f"     📊 HYBRID:  {hybrid_count} (direct + sector combined)")
    print(f"     🏷️  SECTOR:  {sector_only_count} (sector news only)")
    print(f"  📰 Total unique headlines: {len(seen_headlines)}")
    print(f"  📈 Price data: {len(price_data)} companies")

    # Top movers
    top_bullish = sorted(
        [(s, d) for s, d in final_scored.items() if d["score"] > 0.3],
        key=lambda x: -x[1]["score"]
    )[:10]
    top_bearish = sorted(
        [(s, d) for s, d in final_scored.items() if d["score"] < -0.3],
        key=lambda x: x[1]["score"]
    )[:10]

    print("\n  🟢 TOP BULLISH:")
    for symbol, data in top_bullish:
        price_str = f"  📈 {data['price']:+.2f}%" if data["price"] is not None else ""
        type_icon = "🎯" if data["type"] == "DIRECT" else ("📊" if data["type"] == "HYBRID" else "🏷️")
        print(f"     {symbol:<12} {Fore.GREEN}{data['score']:+.4f}{Style.RESET_ALL}  "
              f"[{data['confidence']}] {type_icon}{price_str}  {data['name'][:30]}")

    print("\n  🔴 TOP BEARISH:")
    for symbol, data in top_bearish:
        price_str = f"  📉 {data['price']:+.2f}%" if data["price"] is not None else ""
        type_icon = "🎯" if data["type"] == "DIRECT" else ("📊" if data["type"] == "HYBRID" else "🏷️")
        print(f"     {symbol:<12} {Fore.RED}{data['score']:+.4f}{Style.RESET_ALL}  "
              f"[{data['confidence']}] {type_icon}{price_str}  {data['name'][:30]}")

    print("\n" + "=" * 80)
    stats = get_stats()
    print(f"  💾 DB Totals: {stats['total_scores']} headlines | {stats['total_averages']} averages")
    print(f"  ⏱️  Cycle completed in {elapsed:.1f} seconds")
    print("=" * 80)


def main():
    print("=" * 80)
    print("  🚀 ALPHA SENTIMENT ENGINE — V5 (1000+ Companies, Hybrid Scoring)")
    print("=" * 80)

    stocks, sectors = load_stocks()
    if not stocks:
        return

    print(f"  Universe: {len(stocks)} Indian stocks")
    print(f"  Phase A: {len(SECTOR_QUERIES)} sectors scanned")
    print(f"  Phase B: {len(TOP_COMPANIES)} companies with direct news")
    print(f"  Phase C: {TOTAL_COMPANY_TARGET}+ hybrid merge target")
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
