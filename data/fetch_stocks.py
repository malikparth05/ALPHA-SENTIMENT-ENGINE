#!/usr/bin/env python3
"""
Download the full list of NSE + BSE listed companies with their sectors.
Saves the result as data/indian_stocks.json

This script fetches data from:
1. NSE India (all equity securities)
2. Groups them by industry/sector
"""

import requests
import json
import csv
import io
import time

OUTPUT_FILE = "data/indian_stocks.json"


def fetch_nse_stocks() -> list[dict]:
    """
    Fetch all listed equities from NSE India.
    NSE provides a CSV at their website.
    """
    print("📥 Fetching NSE equity list...")

    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/csv,text/html,application/xhtml+xml",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        reader = csv.DictReader(io.StringIO(response.text))
        stocks = []
        for row in reader:
            symbol = row.get("SYMBOL", "").strip()
            name = row.get("NAME OF COMPANY", "").strip()

            if symbol and name:
                stocks.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": "NSE",
                    "yahoo_ticker": f"{symbol}.NS",
                })

        print(f"   ✅ Found {len(stocks)} NSE stocks")
        return stocks

    except Exception as e:
        print(f"   ⚠️ NSE download failed: {e}")
        print("   Trying alternative source...")
        return []


def fetch_nse_stocks_alternative() -> list[dict]:
    """
    Alternative: Fetch from NSE's JSON API.
    """
    print("📥 Trying NSE JSON API...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    })

    # First hit the main page to get cookies
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)

        # Then fetch the stock listing
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        response = session.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            stocks = []
            for item in data.get("data", []):
                symbol = item.get("symbol", "")
                if symbol:
                    stocks.append({
                        "symbol": symbol,
                        "name": item.get("meta", {}).get("companyName", symbol),
                        "exchange": "NSE",
                        "yahoo_ticker": f"{symbol}.NS",
                        "industry": item.get("meta", {}).get("industry", ""),
                    })
            print(f"   ✅ Found {len(stocks)} stocks from F&O list")
            return stocks
    except Exception as e:
        print(f"   ⚠️ NSE JSON API failed: {e}")

    return []


def fetch_industry_mapping() -> dict[str, str]:
    """
    Try to get industry/sector mapping for NSE stocks.
    Uses NSE's industry listing page.
    """
    print("📥 Fetching industry mapping...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    })

    industries = {}

    # NSE sector indices - we can use these to map stocks to sectors
    sector_indices = [
        "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY AUTO",
        "NIFTY FINANCIAL SERVICES", "NIFTY FMCG", "NIFTY METAL",
        "NIFTY REALTY", "NIFTY ENERGY", "NIFTY INFRASTRUCTURE",
        "NIFTY PSE", "NIFTY MEDIA", "NIFTY PRIVATE BANK",
        "NIFTY COMMODITIES", "NIFTY HEALTHCARE INDEX",
        "NIFTY CONSUMER DURABLES", "NIFTY OIL & GAS",
    ]

    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)

        for index_name in sector_indices:
            try:
                encoded = requests.utils.quote(index_name)
                url = f"https://www.nseindia.com/api/equity-stockIndices?index={encoded}"
                response = session.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    sector = index_name.replace("NIFTY ", "").title()
                    for item in data.get("data", []):
                        symbol = item.get("symbol", "")
                        if symbol and symbol != "NIFTY BANK":
                            industries[symbol] = sector

                time.sleep(0.5)  # Be nice to the API
            except Exception:
                pass

        print(f"   ✅ Got sector mapping for {len(industries)} stocks")
    except Exception as e:
        print(f"   ⚠️ Industry mapping failed: {e}")

    return industries


# Manual overrides for top companies that keyword matching gets wrong
COMPANY_SECTOR_OVERRIDES = {
    # Conglomerates / Holding
    "RELIANCE": "Energy", "ADANIENT": "Infrastructure", "ADANIPORTS": "Logistics",
    "ADANIGREEN": "Energy", "ADANIPOWER": "Energy", "ADANITRANS": "Energy",
    "LT": "Infrastructure", "GRASIM": "Chemicals", "GODREJCP": "FMCG",
    "GODREJPROP": "Real Estate", "GODREJIND": "Chemicals",
    # Banking that keyword might miss
    "BAJFINANCE": "Banking", "BAJAJFINSV": "Banking", "CHOLAFIN": "Banking",
    "SHRIRAMFIN": "Banking", "MUTHOOTFIN": "Banking", "MANAPPURAM": "Banking",
    "PEL": "Banking", "LICHSGFIN": "Banking", "CANFINHOME": "Banking",
    "IDFCFIRSTB": "Banking", "INDUSINDBK": "Banking",
    # IT companies with non-obvious names
    "INFY": "IT", "WIPRO": "IT", "TCS": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "LTTS": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    "MPHASIS": "IT", "TATAELXSI": "IT", "OFSS": "IT", "NAUKRI": "IT",
    "ROUTE": "IT", "HAPPSTMNDS": "IT", "MASTEK": "IT", "SONATA": "IT",
    # Pharma companies
    "CIPLA": "Pharma", "DIVISLAB": "Pharma", "DRREDDY": "Pharma",
    "SUNPHARMA": "Pharma", "LUPIN": "Pharma", "TORNTPHARM": "Pharma",
    "AUROPHARMA": "Pharma", "BIOCON": "Pharma", "ALKEM": "Pharma",
    "MAXHEALTH": "Pharma", "APOLLOHOSP": "Pharma", "LALPATHLAB": "Pharma",
    # Consumer / FMCG
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "DABUR": "FMCG",
    "MARICO": "FMCG", "COLPAL": "FMCG", "EMAMILTD": "FMCG",
    "PATANJALI": "FMCG", "PAGEIND": "Textiles", "TITAN": "Retail",
    "DMART": "Retail", "TRENT": "Retail", "PVRINOX": "Media",
    # Auto
    "TATAMOTORS": "Auto", "M&M": "Auto", "MARUTI": "Auto",
    "EICHERMOT": "Auto", "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto",
    "ASHOKLEY": "Auto", "TVSMOTORS": "Auto", "MOTHERSON": "Auto",
    "BOSCHLTD": "Auto", "EXIDEIND": "Auto", "AMARARAJA": "Auto",
    # Defence / Aerospace
    "HAL": "Defence", "BEL": "Defence", "MAZDOCK": "Defence",
    "COCHINSHIP": "Defence", "GRSE": "Defence",
    # Metals & Mining
    "TATASTEEL": "Metal", "JSWSTEEL": "Metal", "SAIL": "Metal",
    "HINDALCO": "Metal", "VEDL": "Metal", "NMDC": "Metal",
    "JINDALSTEL": "Metal", "NATIONALUM": "Metal", "COALINDIA": "Metal",
    # Energy / Oil & Gas
    "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy", "GAIL": "Energy",
    "NTPC": "Energy", "POWERGRID": "Energy", "TATAPOWER": "Energy",
    "NHPC": "Energy", "IRFC": "Infrastructure", "RECLTD": "Energy",
    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",
    # Cement
    "ULTRACEMCO": "Real Estate", "AMBUJACEM": "Real Estate",
    "SHREECEM": "Real Estate", "ACC": "Real Estate",
    # Insurance
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance", "ICICIPRULI": "Insurance",
    "POLICYBZR": "Insurance",
    # Fintech
    "PAYTM": "Banking", "NYKAA": "Retail", "ZOMATO": "FMCG",
    # Transport
    "IRCTC": "Logistics", "INDIGO": "Logistics",
    # Real Estate
    "DLF": "Real Estate", "OBEROIRLTY": "Real Estate",
    # Chemicals
    "PIDILITIND": "Chemicals", "SRF": "Chemicals", "BERGEPAINT": "Chemicals",
    "ASIANPAINT": "Chemicals",
    # Electronics / Consumer Durables  
    "HAVELLS": "Consumer Durables", "VOLTAS": "Consumer Durables",
    "BLUESTARLT": "Consumer Durables", "CROMPTON": "Consumer Durables",
    "DIXON": "Consumer Durables",
}


def get_sector_from_name(company_name: str, symbol: str = "") -> str:
    """
    Guess the sector from the company name using keywords.
    First checks manual overrides, then uses expanded keyword matching.
    """
    # Check manual overrides first
    if symbol and symbol in COMPANY_SECTOR_OVERRIDES:
        return COMPANY_SECTOR_OVERRIDES[symbol]

    name_lower = company_name.lower()

    sector_keywords = {
        "Banking": ["bank", "finance", "financial", "credit", "lending", "capital", "invest",
                     "fund", "wealth", "asset", "nidhi", "microfinance", "nbfc", "housing fin"],
        "IT": ["tech", "software", "computer", "info", "digital", "cyber", "data", "cloud",
               "system", "solution", "consult", "internet", "e-comm", "online"],
        "Pharma": ["pharma", "drug", "med", "health", "hospital", "bio", "life science",
                    "therapeutic", "diagnos", "laborator", "path lab", "clinic", "care"],
        "Auto": ["motor", "auto", "vehicle", "car", "tyre", "tire", "tractor", "scooter",
                 "bike", "two wheel", "three wheel"],
        "FMCG": ["consumer", "food", "beverage", "dairy", "biscuit", "tea", "coffee", "soap",
                  "personal care", "nutrition", "snack", "spice", "edible", "flour", "rice"],
        "Energy": ["power", "energy", "electric", "solar", "wind", "oil", "gas", "petro",
                    "coal", "renewable", "thermal", "hydro", "nuclear", "refiner"],
        "Metal": ["steel", "iron", "metal", "alumin", "copper", "zinc", "mining", "ore",
                   "alloy", "foundry", "smelt", "casting", "forg"],
        "Telecom": ["telecom", "communication", "mobile", "wireless", "network", "broadband",
                     "fibre", "tower", "satellite"],
        "Real Estate": ["realty", "estate", "housing", "property", "construction", "infra",
                         "build", "cement", "concrete", "ceramics", "tile", "sanitary",
                         "marble", "granite"],
        "Chemicals": ["chem", "fertilizer", "pesticide", "paint", "dye", "pigment", "adhesive",
                       "polymer", "plastic", "resin", "specialty chem", "agrochem", "coating"],
        "Textiles": ["textile", "fabric", "cotton", "garment", "apparel", "silk", "wool",
                      "yarn", "weaving", "spinning", "denim", "fashion"],
        "Media": ["media", "entertainment", "film", "broadcast", "publish", "news", "print",
                   "advertising", "digital media", "content", "animation", "gaming"],
        "Insurance": ["insurance", "assurance", "life insur", "general insur"],
        "Agriculture": ["agri", "seed", "crop", "plantation", "sugar", "farm", "fertili",
                         "irrigation", "horticulture"],
        "Logistics": ["logistics", "transport", "shipping", "warehouse", "cargo", "port",
                       "courier", "express", "supply chain", "aviation", "airline", "railway"],
        "Retail": ["retail", "mart", "store", "shop", "mall", "e-commerce", "jewel",
                    "gold", "diamond", "gem", "ornament", "watch", "luxury"],
        "Hotels": ["hotel", "hospitality", "tourism", "travel", "restaurant", "resort",
                    "catering", "food service"],
        "Paper": ["paper", "packaging", "carton", "pulp", "corrugat", "box", "container"],
        "Defence": ["defence", "defense", "weapon", "ammunition", "aerospace", "shipbuild",
                     "naval", "ordnance", "missile"],
        "Consumer Durables": ["appliance", "electronic", "electrical", "lamp", "light",
                               "fan", "air condition", "refrig", "washing", "kitchen"],
        "Education": ["education", "school", "university", "learning", "coaching", "academy",
                       "training", "skill"],
    }

    for sector, keywords in sector_keywords.items():
        for keyword in keywords:
            if keyword in name_lower:
                return sector

    return "General"


def build_full_stock_list():
    """
    Build the complete stock list with sectors.
    """
    # Step 1: Fetch all NSE stocks
    nse_stocks = fetch_nse_stocks()

    if not nse_stocks:
        nse_stocks = fetch_nse_stocks_alternative()

    if not nse_stocks:
        print("\n❌ Could not fetch stock list from NSE. Using backup approach...")
        # Create a comprehensive list from yfinance
        print("📥 Building stock list from known indices...")
        nse_stocks = build_from_known_lists()

    # Step 2: Try to get industry mapping
    industry_map = fetch_industry_mapping()

    # Step 3: Assign industries to all stocks
    for stock in nse_stocks:
        symbol = stock["symbol"]
        if symbol in industry_map:
            stock["sector"] = industry_map[symbol]
        elif "industry" in stock and stock["industry"]:
            stock["sector"] = stock["industry"]
        else:
            stock["sector"] = get_sector_from_name(stock["name"], symbol)

    # Step 4: Build sector summary
    sectors = {}
    for stock in nse_stocks:
        sector = stock["sector"]
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(stock["symbol"])

    # Step 5: Save
    output = {
        "metadata": {
            "total_stocks": len(nse_stocks),
            "total_sectors": len(sectors),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "exchanges": ["NSE", "BSE"],
        },
        "stocks": nse_stocks,
        "sectors": {sector: symbols for sector, symbols in sorted(sectors.items())},
    }

    import os
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Saved {len(nse_stocks)} stocks to {OUTPUT_FILE}")
    print(f"  📊 {len(sectors)} unique sectors found:")
    for sector, symbols in sorted(sectors.items(), key=lambda x: -len(x[1])):
        print(f"     {sector:<20} → {len(symbols)} companies")
    print(f"{'=' * 60}")


def build_from_known_lists() -> list[dict]:
    """
    Fallback: Build a comprehensive list by downloading from yfinance 
    all tickers that end with .NS or .BO
    """
    import yfinance as yf

    # Get all Nifty indices to cover as many stocks as possible
    indices = [
        "^NSEI",     # Nifty 50
        "^NSMIDCP",  # Nifty Midcap
        "^CNXSC",    # Nifty Smallcap
    ]

    # Known comprehensive list of NSE tickers from major indices
    # This covers Nifty 50 + Next 50 + Midcap 150 + Smallcap 250 = ~500 stocks
    print("   Using known index constituents...")

    # We'll fetch the actual list from BSE's website which is more accessible
    url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Atea=&Status=Active"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.bseindia.com/",
    }

    stocks = []
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                code = item.get("SCRIP_CD", "")
                name = item.get("LONG_NAME", "") or item.get("scrip_name", "")
                nse_symbol = item.get("NSE_SYMBOL", "")

                if name:
                    stock = {
                        "symbol": nse_symbol if nse_symbol else str(code),
                        "name": name,
                        "exchange": "BSE" if not nse_symbol else "NSE+BSE",
                        "bse_code": str(code),
                    }
                    if nse_symbol:
                        stock["yahoo_ticker"] = f"{nse_symbol}.NS"
                    else:
                        stock["yahoo_ticker"] = f"{code}.BO"
                    stocks.append(stock)

            print(f"   ✅ Got {len(stocks)} stocks from BSE")
            return stocks
    except Exception as e:
        print(f"   ⚠️ BSE API failed: {e}")

    return stocks


if __name__ == "__main__":
    print("=" * 60)
    print("  📊 INDIAN STOCK LIST DOWNLOADER")
    print("=" * 60)
    build_full_stock_list()
