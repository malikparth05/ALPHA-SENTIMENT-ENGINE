#!/usr/bin/env python3
# ===========================================
# Alpha Sentiment Engine — Demo Script
# ===========================================
# This is the "test customer" — it sends 3 headlines
# through the async pipeline and prints the results.
#
# BEFORE running this, make sure:
#   1. Redis is running:     docker compose up -d
#   2. Worker is running:    celery -A infrastructure.celery_app worker --loglevel=info
#                            (in a SEPARATE terminal!)
#   3. Then run this:        python demo.py
# ===========================================

import json
import sys

from interface.publisher import publish_news


def main() -> None:
    """Send sample headlines and print the scored results."""

    # ---- 3 test headlines ----
    headlines = [
        ("AAPL", "Apple beats earnings expectations with record iPhone sales"),
        ("TSLA", "Tesla recalls 2 million vehicles over safety concerns"),
        ("MSFT", "Microsoft announces quarterly dividend unchanged"),
    ]

    print()
    print("=" * 60)
    print("  🚀 Alpha Sentiment Engine — Pipeline Demo")
    print("=" * 60)
    print()

    # Step 1: Send all headlines into the queue
    print("📤 Sending headlines to the Redis queue...\n")
    pending = []
    for ticker, headline in headlines:
        result = publish_news(ticker, headline)
        pending.append((ticker, headline, result))
        print(f"   ✅ Queued: [{ticker}] {headline}")

    # Step 2: Wait for the worker to score them
    print()
    print("⏳ Waiting for the Celery worker to score them...\n")

    for ticker, headline, result in pending:
        try:
            # Wait up to 120 sec (first run downloads model = slow)
            sentiment = result.get(timeout=120)

            # Pretty-print the JSON result
            print(json.dumps(sentiment, indent=2))
            print()

        except Exception as e:
            print(f"   ❌ Error scoring [{ticker}]: {e}", file=sys.stderr)

    print("=" * 60)
    print("  ✅ Pipeline demo complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
