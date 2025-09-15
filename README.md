# OSTA Stock News Sentiment — README

## What this script does

`osta_sentiment.py` fetches **stock news headlines from Finviz**, scores them with a **finance-tuned sentiment model (FinBERT)** or **VADER** as fallback, **de-duplicates**, applies **recency weighting**, and aggregates into **daily signals** with optional **plots**, **CSV export**, and a **quick diagnostic backtest**.

**Pipeline**

1. Fetch Finviz news pages per ticker (polite retries + optional on-disk cache).
2. Parse headlines + timestamp (assumes US/Eastern → stored in UTC).
3. De-duplicate repeated/syndicated titles per ticker.
4. Score sentiment (FinBERT by default; VADER fallback).
5. Weight by recency (exponential decay with configurable half-life).
6. Aggregate by day → rolling z-score normalization (no look-ahead).
7. Save CSVs, generate plots, and (optionally) run a simple next-day return check.

---

## Install

```bash
# Core
pip install requests beautifulsoup4 pandas numpy matplotlib

# Recommended
pip install seaborn requests-cache python-dateutil pytz tqdm

# FinBERT (for best results)
pip install torch torchvision torchaudio transformers

# Optional backtest
pip install yfinance

# Optional word cloud
pip install wordcloud
```

---

## Usage

```bash
# Basic
python osta_sentiment.py --tickers AAPL MSFT GOOGL --export --plots

# With date range (YYYY-MM-DD)
python osta_sentiment.py --tickers TSLA --start 2025-08-15 --end 2025-09-15 --export --plots

# Use VADER instead of FinBERT
python osta_sentiment.py --tickers NVDA --sentiment-model vader --export

# Run simple diagnostics/backtest (needs yfinance)
python osta_sentiment.py --tickers AAPL MSFT --backtest --export
```

**Common flags**

* `--tickers T1 T2 ...` (required)
* `--start / --end` filter dates (interpreted in ET for parsing; stored UTC)
* `--export` write CSVs to `./StockSentiment/results/`
* `--plots` save PNGs to `./StockSentiment/plots/`
* `--no-cache` disable HTTP caching (default cache TTL 30 min; change with `--cache-ttl 900`)
* `--half-life 24` recency half-life (hours)
* `--rolling-window 20` normalization window (days)
* `--sentiment-model finbert|vader`
* `--backtest` quick toy long/short diagnostic

Run `-h` for the full list.

---

## Outputs

* **CSV** (in `StockSentiment/results/`):

  * `news_data_*.csv`: per-headline with timestamps + sentiment scores.
  * `daily_sentiment_*.csv`: daily aggregates (`avg_compound_w`, `normalized_score`, `news_count`, etc.).
* **Plots** (in `StockSentiment/plots/`):

  * `{TICKER}_sentiment_norm.png` — rolling z-score over time.
  * `{TICKER}_news_volume.png` — daily article counts.
  * `sentiment_heatmap.png`, `comparative_sentiment.png`, `comparative_volume.png`.
  * `{TICKER}_wordcloud.png` (if `wordcloud` installed).

---

## Notes & Tips

* **FinBERT** gives better finance domain signal; install `transformers` + `torch` to enable. Otherwise script auto-falls back to **VADER**.
* **Caching**: uses `requests-cache` if installed; reduces rate-limit risk and speeds up repeated runs.
* **Source**: this script scrapes Finviz HTML; be polite and expect occasional throttling. If you see 403/429, re-try later or reduce ticker count.
* **Evaluation**: `--backtest` is a **toy** diagnostic (not investment advice).
* **Limitations**: headlines ≠ articles; timestamps are parsed best-effort; scraping markup can change.

---

## Project layout (created on first run)

```
StockSentiment/
  logs/          # rotating log file
  debug/         # raw HTML snapshots
  cache/         # parsed JSON cache buckets
  results/       # CSV outputs
  plots/         # PNG charts
```

That’s it—point it at tickers, flip on `--export`/`--plots`, and iterate on the CSVs/figures as features for OSTA.
