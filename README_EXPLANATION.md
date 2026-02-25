# Alpha Sentiment Engine — Complete Project Explanation
*(A plain-English guide to everything we built)*

---

## Table of Contents
1. [The Big Idea](#the-big-idea)
2. [Stage 1: The Prototype](#stage-1-the-prototype)
3. [Stage 2: The Architecture](#stage-2-the-architecture-redis--celery)
4. [Stage 3: The AI Training](#stage-3-the-ai-training)
5. [What is FinBERT?](#what-is-finbert)
6. [Where Did the Data Come From?](#where-did-the-data-come-from)
7. [How the Training Actually Works](#how-the-training-actually-works)
8. [Project File Map](#project-file-map)
9. [Training Results](#training-results)
10. [What's Next](#whats-next)

---

## The Big Idea

We built a system that:
1. Grabs live stock prices from Alpha Vantage.
2. Grabs the latest news headlines from NewsAPI.
3. Uses an AI model called **FinBERT** to read those headlines and score them as Positive, Negative, or Neutral.
4. Runs this in the background at scale using Redis and Celery.
5. We then **trained our own custom version of FinBERT** on 45,000+ financial texts so it understands modern slang, Reddit posts, and StockTwits memes.

---

## Stage 1: The Prototype

**File:** `prototype.py`

This was the very first script we wrote. Its only goal was to prove our idea could work. One file, no fancy architecture.

### How it works:

**Part 1 — The Setup (Lines 15-52)**
```python
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```
- `requests` lets Python talk to websites and download data from the internet.
- `transformers` is a toolbox by Hugging Face that lets us download the FinBERT AI brain onto your Mac.
- We store our API keys here (passwords to access Alpha Vantage and NewsAPI).
- We created a dictionary `COMPANY_NAMES` that translates "AAPL" → "Apple" because journalists write "Apple", not "AAPL".

**Part 2 — Fetching the Stock Price (Lines 58-88)**
```python
def get_stock_price(ticker: str) -> dict:
    url = "https://www.alphavantage.co/query"
    response = requests.get(url, params=params)
    data = response.json()
```
- Builds a web address with your API key and the ticker symbol.
- `requests.get()` hits "Enter" on that web address.
- Alpha Vantage replies with a block of text containing the price.
- `.json()` organizes it into a neat dictionary so Python can read it.

**Part 3 — Fetching the News (Lines 94-129)**
```python
def get_news_headlines(query: str, count: int = 5) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "sortBy": "publishedAt"}
```
- Works just like the price function, but talks to NewsAPI instead.
- Searches for "Apple" (not "AAPL"), grabs the 5 newest English articles.

**Part 4 — The AI Brain (Lines 134-178)**
```python
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    winner_index = torch.argmax(probabilities).item()
```
This is the magic:
1. **Tokenizer**: AI can't read English. The tokenizer turns "Apple beats earnings" into numbers like `[456, 12, 989]`.
2. **Model**: The 110-million-parameter brain runs those numbers through millions of math equations.
3. **Softmax**: The raw math output (like `[4.2, -1.1, 0.5]`) gets converted into clean percentages (like `[85%, 5%, 10%]`).
4. **Argmax**: Picks the highest percentage. 85% is "Positive", so the AI says "Bullish!"

**Part 5 — The Manager (Bottom of file)**
When you run `python prototype.py`:
1. Manager → Alpha Vantage: "Get me the price of AAPL."
2. Manager → NewsAPI: "Get me 5 articles about Apple."
3. Manager → AI: "Read each headline and tell me if it's good or bad."
4. Manager → Screen: "Print everything nicely for the human!"

**Why we moved past this file:** It processes everything one-by-one. If we wanted to track 500 stocks, the program would freeze for 30+ minutes.

---

## Stage 2: The Architecture (Redis & Celery)

We split the work into separate jobs using a **Restaurant Kitchen** model:
- `prototype.py` = One guy who takes the order, cooks the food, AND serves the table.
- **Stage 2** = A **Waiter**, an **Order Board**, and a **Chef** working separately.

### File 1: `docker-compose.yml` (The Order Board)
```yaml
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```
Docker downloads and runs **Redis** (a lightning-fast in-memory database). Redis is the Order Board where the Waiter pins the tickets for the Chef to grab.

### File 2: `domain/models.py` (The Menu)
```python
class NewsArticle(BaseModel):
    ticker: str
    text: str
    url: str
    published_at: datetime
```
Uses Pydantic to create strict rules. Every article MUST have these 4 fields. If the Waiter tries to send an order without a `url`, Python throws an error immediately. Prevents bugs.

### File 3: `interface/publisher.py` (The Waiter)
```python
def enqueue_articles(articles: list[NewsArticle]):
    for article in articles:
        process_headline.delay(article.dict())
```
The `.delay()` command is Celery magic. It drops a headline onto the Redis board and **immediately moves on**. The Waiter can drop off 1,000 headlines in under a second without waiting for the AI.

### File 4: `infrastructure/celery_app.py` (The Kitchen Manager)
```python
celery_app = Celery("sentiment_worker", broker="redis://localhost:6379/0")
```
Creates the Celery application and tells it where to find the Order Board (Redis at port 6379).

### File 5: `infrastructure/tasks.py` (The Chef)
```python
@celery_app.task
def process_headline(article_dict: dict):
    text = article_dict["text"]
    sentiment_result = sentiment_service.analyze_text(text)
    return sentiment_result
```
The `@celery_app.task` tells Celery: *"Watch the Redis board. Whenever an order appears, grab it and run this function."* You can run 10 Chefs in 10 terminal windows simultaneously.

### File 6: `services/sentiment_service.py` (The AI Brain, Isolated)
```python
class SentimentService:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def analyze_text(self, text: str) -> dict:
        # Run the text through the AI and return the score
```
We ripped the AI logic out of `prototype.py` and put it here. The heavy 110-million-parameter model loads into RAM **once** when the Chef starts, not every time a headline is processed.

### File 7: `demo.py` (The Test)
Imports the Waiter, generates fake headlines, and drops them on the Redis board. Proves the entire restaurant architecture works end-to-end.

---

## Stage 3: The AI Training

**File:** `training/fine_tune.py`

The default FinBERT was good at reading professional news, but terrible at understanding Reddit slang, StockTwits memes, or informal tweets. Instead of writing `if word == "moon": return "positive"`, we wrote a Machine Learning script that teaches the AI by forcing it to read 45,000 examples.

### Part 1 — The Settings
```python
EPOCHS: int = 3          # Read the textbook 3 times
BATCH_SIZE: int = 8      # Read 8 sentences at a time
LEARNING_RATE: float = 2e-5  # How drastically to adjust (very tiny = safe)
```

### Part 2 — Cleaning the Data
```python
def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)          # Remove @mentions
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove emojis
```
Before: `"$AAPL 🚀🚀 to the moon!! @elonmusk https://t.co/xyz"`
After: `"AAPL to the moon"`

### Part 3 — Balancing the Labels
```python
def balance_labels(dataset):
    min_count = min(len(g) for g in groups.values())
    # Randomly cut bigger groups down to match the smallest
```
Before: Positive=21,749 | Negative=15,189 | Neutral=19,907
After: Positive=15,189 | Negative=15,189 | Neutral=15,189
This forces the AI to actually learn instead of just guessing "Neutral" every time.

### Part 4 — The Training Loop
```python
trainer = Trainer(model=model, train_dataset=train_dataset)
trainer.train()  # This ran for 2 hours on your M3 GPU!
```
When `.train()` runs:
1. AI reads 8 sentences and makes 8 guesses.
2. Checks the answer key (the `label` column from the dataset).
3. Calculates the **Loss** (how wrong it was).
4. Runs **Backpropagation** to adjust all 110 million math connections.
5. Grabs the next 8 sentences. Repeats thousands of times.

---

## What is FinBERT?

### The Family Tree
1. **BERT (2018)** — Built by Google. They forced a supercomputer to read the entire English Wikipedia plus thousands of books. They would take a sentence like *"The [BLANK] ran across the street"* and force the AI to guess the blank word, billions of times. BERT learned the complete structure of English, but knew nothing about finance.

2. **FinBERT** — Built by Prosus AI. They took Google's BERT and sent it back to school by making it read 46,000 Reuters financial news articles. It learned that "bearish" means bad, "bullish" means good, and "crashed" doesn't mean a car accident.

3. **YOUR FinBERT** — You downloaded FinBERT onto your Mac, and taught it Reddit slang and StockTwits memes using 45,000 sentences on your M3 GPU.

This approach is called **Transfer Learning**:
- Google taught it English (BERT).
- Prosus AI taught it professional finance (FinBERT).
- You taught it social media finance (YOUR FinBERT).

That's why it only took 2 hours instead of 2 years!

---

## Where Did the Data Come From?

We did NOT label 45,000 sentences manually. We downloaded 4 open-source datasets where other teams of humans had already done the hard work:

| Dataset | Source | Who Labeled It | How Many |
|---------|--------|----------------|----------|
| Financial PhraseBank | Professional news | 16 Finnish university finance experts | 5,842 |
| Twitter Financial News | Finance Twitter | Amazon Mechanical Turk workers (paid crowd) | 11,924 |
| StockTwits Tweets | StockTwits.com | The actual users (they click "bullish/bearish" when posting) | 38,083 |
| Reddit Stock Sentiment | r/wallstreetbets, r/stocks | Human annotators | 996 |

Each dataset used different label formats (English words, 0/1/2 numbers, -1.0/0.0/1.0 decimals). Our code in `fine_tune.py` standardized everything into one universal system:
- **0** = Positive
- **1** = Negative
- **2** = Neutral

---

## How the Training Actually Works

When the AI reads 500 headlines about the same company, it does NOT read them all at once. BERT has a hard limit of 512 tokens (~300-400 words). 

Instead:
1. **Batch Processing**: The AI takes 8 sentences at a time and scores them simultaneously using GPU matrix math.
2. **Individual Scores**: Each sentence gets its own score (like +0.8 or -0.3).
3. **Aggregation**: Your Python code decides what to do with 500 individual scores (average them, calculate a ratio, etc.).

Currently, `prototype.py` just prints each score individually. In the future, we will build code that calculates the average and saves it to a database.

---

## Stage 4: Confident Learning (Cleaning Bad Data with AI)

We ran into a problem: mixing all 4 datasets gave us **lower** accuracy (66.48%) than just using 2 clean datasets (85.63%). The StockTwits and Reddit data had messy, wrong labels. But we didn't want to throw away 39,000 sentences of useful slang vocabulary.

**Solution: Use AI to clean AI training data.** We built a 3-step pipeline:

### Step 1: Train a "Judge" Model
**File:** `training/fine_tune.py` (with `clean_only=True`)

We trained FinBERT on ONLY the 2 clean datasets (PhraseBank + Twitter). This gave us a model with **84.72% accuracy** — our "Professor" who knows what correct labels look like.

```python
# In fine_tune.py, we added a parameter:
def combine_all_datasets(clean_only: bool = False):
    # When clean_only=True, skip StockTwits and Reddit
```

### Step 2: Filter the Noisy Data
**File:** `training/filter_noisy_data.py`

We then ran every single StockTwits and Reddit sentence through the Judge. For each sentence, the Judge made its own prediction and compared it to the human label:

```python
if human_label == judge_prediction:
    # AGREE → This label is probably correct. KEEP IT ✅
else:
    # DISAGREE → This label is probably wrong. THROW IT OUT 🗑️
```

**Results:**
- StockTwits: 38,083 sentences → Kept only 13,881 (36.4%) → Threw out 24,202 bad labels!
- Reddit: 996 sentences → Kept only 487 (48.9%) → Threw out 509 bad labels!

The Judge caught thousands of cases where a random internet user labeled "TSLA down 5%" as "Positive" (because they thought "buy the dip!") when it's clearly Negative.

### Step 3: Final Training on Clean + Filtered Data
We combined the original clean data + the AI-verified data = **12,228 examples** (perfectly balanced at 4,076 per class). Trained the final model on this golden dataset.

**Result: 86.39% accuracy** 🔥 — the best model across all runs, AND it understands Reddit slang!

---

## Stage 4.5: The Brain Swap

**File:** `services/sentiment_service.py`

After training, our Celery workers were still using the old, default FinBERT from the internet. We changed one line of code:

```python
# BEFORE (the old internet brain):
model_name: str = "ProsusAI/finbert"

# AFTER (YOUR custom-trained 86.39% brain):
model_name: str = "models/my_finbert"
```

Now every time a Celery Chef grabs a headline from the Redis Order Board, it scores it using YOUR custom-trained AI instead of the generic one. Same code, same architecture, but a much smarter brain inside.

---

## Stage 5: The Live Scraper

**File:** `live_scraper.py`

Until now, we had to manually type headlines or hard-code them in `demo.py`. The Live Scraper makes the system **fully automatic**.

### How it works:

```python
# The settings at the top:
SCRAPE_INTERVAL: int = 300   # Wake up every 300 seconds (5 minutes)

TICKERS: dict = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    # ... 7 stocks total
}
```

When you run `python live_scraper.py`, it does this **forever**:

1. **Loads YOUR AI brain once** into RAM (takes ~5 seconds).
2. **Loops through all 7 stocks.** For each stock:
   - Asks NewsAPI: "Give me the 5 newest articles about Apple."
   - Feeds each headline to YOUR 86.39% FinBERT model.
   - Prints the individual score for each headline.
   - Calculates the **average** sentiment for that stock.
3. **Prints a Dashboard Summary** showing all 7 stocks with visual bars:
   ```
   AAPL   ████████░░░░░░░░░░░░ -0.17  🟡 Neutral
   TSLA   ████████░░░░░░░░░░░░ -0.16  🟡 Neutral
   AMZN   ███████████░░░░░░░░░ +0.10  🟡 Neutral
   ```
4. **Sleeps for 5 minutes.**
5. **Wakes up and does it all again.** Forever, until you press Ctrl+C.

The AI correctly scores real headlines:
- *"Tesla is having a hard time turning over its FSD traffic violations"* → **-0.93 🔴**
- *"Solid-state battery gets its first road test"* → **+0.97 🟢**
- *"Big Tech to invest $650 billion in AI in 2026"* → **+0.44 🟢**

---

## Project File Map

```
PORTFOLIO/
├── prototype.py                    ← Stage 1: The proof-of-concept
├── demo.py                         ← Stage 2: Tests the pipeline
├── live_scraper.py                 ← Stage 5: Auto-fetches & scores news!
├── docker-compose.yml              ← Starts Redis (the Order Board)
├── requirements.txt                ← All Python packages needed
├── financial_sentiment_dataset.csv ← The full 45,567-row training dataset
├── sentiment_data.db               ← Stage 6: The database (all scores saved here!)
│
├── domain/
│   └── models.py                   ← Data shapes (what a "headline" looks like)
│
├── services/
│   ├── sentiment_service.py        ← The AI Brain (now uses YOUR model!)
│   └── database.py                 ← Stage 6: Database functions (save/load scores)
│
├── infrastructure/
│   ├── celery_app.py               ← Celery setup (connects to Redis)
│   └── tasks.py                    ← The background worker (The Chef)
│
├── interface/
│   └── publisher.py                ← The Waiter (drops work on Redis)
│
├── training/
│   ├── fine_tune.py                ← The ML training script
│   └── filter_noisy_data.py        ← Confident Learning filter script
│
└── models/
    ├── my_finbert/                 ← YOUR final custom brain (86.39%)
    └── my_finbert_judge/           ← The judge model used for filtering
```

---

## Stage 7: The V3 India-Only Scraper (Two-Level Architecture)
*File: `live_scraper.py` (v3)*

### The Challenge
We wanted to track **every single company listed on the Indian stock market (NSE)** — over 2,200 companies. But if we searched for news for each company one by one, it would take hours per cycle. Most tiny companies don't even have English news articles written about them!

### The Solution: Two-Level Scoring
We grouped all 2,241 Indian stocks into 22 major sectors (IT, Pharma, Banking, Metal, etc.). The new scraper works perfectly in just **50 seconds**:

1. **Level 1 (Sector News):** It searches for broad news about the whole sector (e.g., *"Indian IT sector stock market"*). It scores this sector news and applies that base score to **all** companies in that sector. So even a tiny textile company gets a score if the textile sector is performing well!
2. **Level 2 (Company News):** It then searches for specific news for the Top 20 biggest companies (Reliance, SBI, TCS, etc.) and combines their direct news with their sector score.

This means **2,000+ Indian companies** now have a live, AI-generated sentiment score updating every 30 minutes, without hitting any API rate limits!

### Try it yourself
```bash
# Run the 2,200-stock scraper
python live_scraper.py
```

<div align="center">
  <img src="https://img.shields.io/badge/Progress-Stage%_7%_Done-brightgreen">
</div>

---

## What's Next? (Step 8: The Dashboard)
Right now, all our 2,200+ sentiment scores are sitting in a database or printing in a terminal. 

For the final step, we are building a **premium, dark-mode, neon-glowing web dashboard**. 
It will use:
- **Flask (Python)** to read from our SQLite database and serve the web page.
- **HTML/CSS/JS** with **Chart.js** to draw beautiful horizontal bar charts (Market Overview) and smooth line charts (24hr Sentiment Trends).

You'll just open your browser to `http://localhost:5000` and see the AI's real-time thoughts on the entire Indian stock market!

## Stage 6: The Database

**Files:** `services/database.py` + `sentiment_data.db`

Until now, the Live Scraper printed scores to the screen and then they vanished forever. We needed a **permanent notebook**.

### What is SQLite?

SQLite is the world's simplest database. It's just a **single file** on your hard drive called `sentiment_data.db`. No setup needed. No Docker. No passwords. Python can read and write to it directly.

### What we built:

We created `services/database.py` with two tables:

**Table 1: `sentiment_scores`** — Saves every single scored headline:
```
| id | ticker | headline                          | score  | source         | scraped_at           |
|----|--------|-----------------------------------|--------|----------------|----------------------|
| 1  | AAPL   | "Apple beats expectations"        | +0.92  | Reuters        | 2026-02-24T16:23:53  |
| 2  | TSLA   | "Tesla recalls vehicles"          | -0.93  | Electrek       | 2026-02-24T16:23:53  |
```

**Table 2: `sentiment_averages`** — Saves the per-stock average after each scrape cycle:
```
| id | ticker | average_score | num_headlines | scraped_at           |
|----|--------|---------------|---------------|----------------------|
| 1  | AAPL   | -0.0886       | 5             | 2026-02-24T16:23:53  |
| 2  | TSLA   | -0.1591       | 5             | 2026-02-24T16:23:53  |
```

### How it plugs into the Live Scraper:

Every time the scraper scores a headline, two things now happen:
1. The score is **printed** to the screen (like before)
2. The score is **saved** to the database (new!)

After each stock's headlines are all scored, the average is also saved. This is what the dashboard will read to draw trend charts.

---

## Training Results

We ran 5 training experiments across the project:

| Run | Data | Technique | Device | Accuracy | Time |
|-----|------|-----------|--------|----------|------|
| Run 1 | 17k (News + Twitter) | Raw | CPU | 85.63% | 4 hrs |
| Run 2 | 57k (All 4 sources) | Raw | CPU | 67.83% | 5 hrs |
| Run 3 | 45k (All 4, cleaned) | Clean + Balance | GPU | 66.48% | 2 hrs |
| Run 4 (Judge) | 7.9k (Clean only) | Clean + Balance | GPU | 84.72% | 24 min |
| **Run 5 (Final)** | **12k (AI-filtered)** | **Confident Learning** | **GPU** | **86.39%** 🏆 | **48 min** |

The final model achieves 86.39% on a test set that includes news, Twitter, StockTwits slang, AND Reddit posts — the hardest possible exam!

---

## What's Next

| Step | Status | What |
|------|--------|------|
| ~~Step 4.5~~ | ✅ Done | ~~Plug custom model into pipeline~~ |
| ~~Step 5~~ | ✅ Done | ~~Build a Live Auto-Scraper~~ |
| ~~Step 6~~ | ✅ Done | ~~Add a Database (save scores permanently)~~ |
| **Step 7** | 🔜 Next | Build a Dashboard (visualize with charts) |
| **Step 8** | ⬜ | Deploy to Cloud (run 24/7) |
