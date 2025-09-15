
"""
OSTA Stock News Sentiment

- Robust date parsing (handles ET timezone and time-only rows)
- Requests caching (polite scraping with TTL, retries, and logging)
- Safer HTML parsing with multiple selectors + defensive checks
- Optional FinBERT sentiment (finance-tuned) with VADER fallback
- De-duplication of headlines and optional recency weighting
- Rolling z-score normalization to avoid look-ahead bias
- Comparative plots and word clouds saved to disk
- CLI via argparse (no Jupyter/ipywidgets)
- Structured logging with rotating file handler
- Optional basic diagnostic eval (correlation/backtest) if yfinance available

Usage (examples):
    python osta_sentiment.py --tickers AAPL MSFT GOOGL --start 2025-08-15 --end 2025-09-15 --sentiment-model finbert --export --plots
    python osta_sentiment.py --tickers TSLA --no-plots --no-export
    python osta_sentiment.py --tickers AAPL MSFT --backtest
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

# --- Third-party imports (all optional except requests/bs4/pandas/numpy/matplotlib) ---
import requests
import requests.adapters
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Matplotlib/Seaborn (plots are optional)
import matplotlib
matplotlib.use("Agg")  # headless by default
import matplotlib.pyplot as plt

with contextlib.suppress(ImportError):
    import seaborn as sns

# NLTK (VADER fallback)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# WordCloud (optional)
with contextlib.suppress(ImportError):
    from wordcloud import WordCloud

# Caching
with contextlib.suppress(ImportError):
    import requests_cache

# FinBERT (optional)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except Exception:
    FINBERT_AVAILABLE = False

# yfinance for optional diagnostics/backtest
with contextlib.suppress(ImportError):
    import yfinance as yf

# Date parsing/timezones
from dateutil import parser as dtparser
import pytz

# Progress bars
with contextlib.suppress(ImportError):
    from tqdm import tqdm
    TQDM_AVAILABLE = True
if 'tqdm' not in globals():
    # Fallback dummy tqdm
    def tqdm(it, **kwargs):
        return it
    TQDM_AVAILABLE = False


# =========================
# Configuration & Logging
# =========================

DEFAULT_BASE_DIR = os.path.join(".", "StockSentiment")
ET = pytz.timezone("US/Eastern")
UTC = pytz.UTC

def setup_logging(base_dir: str, verbose: bool = True) -> None:
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    log_path = os.path.join(base_dir, "logs", "osta_sentiment.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file
    fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logging.info("Logging initialized. Log file: %s", log_path)


# =========================
# Helpers
# =========================

def ensure_nltk_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

def robust_session(cache_name: Optional[str], cache_expire_seconds: int) -> requests.Session:
    """
    Build a requests session with retries; optionally install requests_cache if available.
    """
    if 'requests_cache' in sys.modules and cache_name:
        # Install cache globally
        requests_cache.install_cache(cache_name, backend="sqlite", expire_after=cache_expire_seconds)
        logging.info("requests_cache enabled: %s (TTL=%ss)", cache_name, cache_expire_seconds)

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def normalize_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_finviz_datetime(raw_date: str, raw_time: str) -> datetime:
    """
    Finviz rows can show date and time; sometimes only time for continuing rows.
    We interpret times as US/Eastern and return timezone-aware UTC datetime.
    """
    raw_date = (raw_date or "").strip()
    raw_time = (raw_time or "").strip()
    now_et = datetime.now(ET)

    # Finviz uses formats like "Sep-12-25 08:31AM" OR (only time) "09:12AM"
    if raw_date == "" or re.match(r"^\d{1,2}:\d{2}\s?(AM|PM)$", raw_date, re.IGNORECASE):
        # Only time was present; treat raw_date as actually time; set date to today ET
        t = raw_date if raw_date else raw_time
        base_date = now_et.date()
        dt_local = dtparser.parse(f"{base_date} {t}")
        dt_et = ET.localize(dt_local) if dt_local.tzinfo is None else dt_local.astimezone(ET)
    else:
        # We have a date token; Finviz shows %b-%d-%y or %b-%d (no year)
        try:
            candidate = dtparser.parse(f"{raw_date} {raw_time}", dayfirst=False, yearfirst=False, fuzzy=True)
        except Exception:
            # Try date only; time-only later
            candidate = dtparser.parse(raw_date, dayfirst=False, yearfirst=False, fuzzy=True)
        if candidate.tzinfo is None:
            dt_et = ET.localize(candidate)
        else:
            dt_et = candidate.astimezone(ET)

    return dt_et.astimezone(UTC)


# =========================
# Sentiment Models
# =========================
# --- at top of file (imports) ---
from packaging import version as _pkg_version

class FinBertScorer:
    """
    Finance-specific sentiment using ProsusAI/finbert (safetensors only).
    Scores return dict with pos/neg/neu and 'compound' = pos - neg
    """
    def __init__(self, model_id: str = "ProsusAI/finbert"):
        if not FINBERT_AVAILABLE:
            raise RuntimeError("FinBERT not available. Install transformers+torch+safetensors to use this model.")

        # Security: require torch >= 2.6 for any non-safetensors model files.
        # We will attempt safetensors-only load and bail out if torch is old.
        if _pkg_version.parse(torch.__version__) < _pkg_version.parse("2.6"):
            logging.warning("Detected torch %s (<2.6). Loading FinBERT only via safetensors; "
                            "otherwise will fall back to VADER.", torch.__version__)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Prefer/force safetensors; if not present this will raise in older repos
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                use_safetensors=True
            )
            self.model.eval()
            logging.info("Loaded FinBERT (safetensors) model: %s", model_id)
        except Exception as e:
            # Do not try to load .bin with old torch; safer to fall back.
            raise RuntimeError(
                f"FinBERT safetensors load failed ({e}). "
                f"Upgrade torch>=2.6 and ensure safetensors files exist for {model_id}."
            )

    @torch.inference_mode()
    def score(self, texts: List[str]) -> List[dict]:
        if not texts:
            return []
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # finbert classes: [neutral, positive, negative]
        out = []
        for p in probs:
            out.append({"pos": float(p[1]), "neg": float(p[2]), "neu": float(p[0]), "compound": float(p[1] - p[2])})
        return out


class VaderScorer:
    def __init__(self):
        ensure_nltk_vader()
        self.sia = SentimentIntensityAnalyzer()

    def score(self, texts: List[str]) -> List[dict]:
        out = []
        for t in texts:
            s = self.sia.polarity_scores(t or "")
            out.append({
                "pos": float(s.get("pos", 0.0)),
                "neg": float(s.get("neg", 0.0)),
                "neu": float(s.get("neu", 0.0)),
                "compound": float(s.get("compound", 0.0)),
            })
        return out


# =========================
# Analyzer
# =========================

@dataclass
class AnalyzerConfig:
    tickers: List[str]
    base_dir: str = DEFAULT_BASE_DIR
    use_cache: bool = True
    cache_ttl_seconds: int = 60 * 30  # 30 minutes
    finviz_base: str = "https://finviz.com/quote.ashx?t="
    user_agent: str = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    sentiment_model: str = "finbert"  # 'finbert' or 'vader'
    half_life_hours: float = 24.0
    rolling_window: int = 20
    export_results: bool = True
    make_plots: bool = True
    show_plots: bool = False  # ignored in headless; kept for completeness
    min_articles_per_day_flag: int = 3
    backtest: bool = False
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class StockSentimentAnalyzer:
    def __init__(self, cfg: AnalyzerConfig):
        self.cfg = cfg
        self.session = robust_session(
            cache_name="finviz_cache" if cfg.use_cache else None,
            cache_expire_seconds=cfg.cache_ttl_seconds,
        )
        self.headers = {"User-Agent": cfg.user_agent}
        self.news_df: Optional[pd.DataFrame] = None
        self.daily_sentiment_df: Optional[pd.DataFrame] = None
        self.summarized: Dict[str, any] = {}

        # Sentiment model selection
        if cfg.sentiment_model.lower() == "finbert":
            if FINBERT_AVAILABLE:
                try:
                    self.scorer = FinBertScorer()
                except Exception as e:
                    logging.warning("FinBERT unavailable (%s). Falling back to VADER.", e)
                    self.scorer = VaderScorer()
            else:
                logging.warning("FinBERT libs not available; using VADER.")
                self.scorer = VaderScorer()
        else:
            self.scorer = VaderScorer()

        # Folders
        for sub in ("cache", "results", "plots", "debug"):
            os.makedirs(os.path.join(cfg.base_dir, sub), exist_ok=True)

    # ------------- Fetch & Parse -------------
    def _fetch_html_for_ticker(self, ticker: str) -> Optional[str]:
        url = f"{self.cfg.finviz_base}{ticker}"
        try:
            resp = self.session.get(url, headers=self.headers, timeout=15)
            txt = resp.text or ""
            if "Access denied" in txt or resp.status_code in (403, 429):
                logging.error("%s: Access denied/throttled by Finviz (status=%s).", ticker, resp.status_code)
                return None
            # Save debug HTML occasionally
            debug_path = os.path.join(self.cfg.base_dir, "debug", f"{ticker}_{int(time.time())}.html")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(txt)
            return txt
        except Exception as e:
            logging.exception("Error fetching %s: %s", ticker, e)
            return None

    @staticmethod
    def _extract_rows_from_html(html: str) -> List[Dict[str, str]]:
        """
        Returns rows with keys: title, raw_date, raw_time
        Finviz puts date/time in first td; when multiple headlines share same date,
        subsequent rows show only time.
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("#news-table") or soup.select_one("table.news-table")
        if not table:
            # Try broader fallback
            candidates = soup.select("table")
            table = None
            for c in candidates:
                if "News" in c.get_text(" ", strip=True)[:500]:
                    table = c
                    break
        if not table:
            return []

        rows = []
        last_date_token = ""
        for tr in table.select("tr"):
            a = tr.find("a")
            td = tr.find("td")
            if not a or not td:
                continue
            title = a.get_text(strip=True)
            dt_token = td.get_text(strip=True)

            # dt_token may be "Sep-12-25 08:31AM" or just "09:12AM"
            parts = dt_token.split()
            if len(parts) == 1:
                raw_date = last_date_token
                raw_time = parts[0]
            else:
                raw_date = parts[0]
                raw_time = parts[1]
                last_date_token = raw_date

            if title and raw_time:
                rows.append({
                    "title": title,
                    "raw_date": raw_date,
                    "raw_time": raw_time,
                })
        return rows

    def fetch_all_news(self) -> Dict[str, List[Dict[str, str]]]:
        news = {}
        tickers = list(dict.fromkeys([t.upper().strip() for t in self.cfg.tickers if t.strip()]))
        if not tickers:
            return news

        logging.info("Fetching news for %d tickers...", len(tickers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(2, len(tickers)))) as ex:
            futs = {ex.submit(self._fetch_html_for_ticker, t): t for t in tickers}
            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="Fetching"):
                t = futs[fut]
                html = fut.result()
                if not html:
                    continue
                rows = self._extract_rows_from_html(html)
                news[t] = rows
                logging.info("%s: parsed %d rows", t, len(rows))
                # Persist parsed rows to JSON (time-bucketed)
                bucket = int(time.time() // self.cfg.cache_ttl_seconds) if self.cfg.cache_ttl_seconds else int(time.time())
                cache_file = os.path.join(self.cfg.base_dir, "cache", f"{t}_{bucket}.json")
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False)
        return news

    # ------------- Process/Transform -------------
    def process_news_data(self, raw: Dict[str, List[Dict[str, str]]]) -> pd.DataFrame:
        data = []
        for ticker, rows in raw.items():
            for r in rows:
                raw_date = r.get("raw_date", "")
                raw_time = r.get("raw_time", "")
                try:
                    published = parse_finviz_datetime(raw_date, raw_time)
                except Exception:
                    # fallback: use ET now with provided time if possible
                    try:
                        t_only = dtparser.parse(raw_time)
                        published = ET.localize(datetime.combine(datetime.now().date(), t_only.time())).astimezone(UTC)
                    except Exception:
                        published = datetime.now(UTC)

                data.append({
                    "ticker": ticker,
                    "title": r.get("title", "").strip(),
                    "raw_date": raw_date,
                    "raw_time": raw_time,
                    "published_at_utc": published,
                })
        if not data:
            return pd.DataFrame(columns=["ticker", "title", "published_at_utc"])

        df = pd.DataFrame(data)
        # Optional date filtering
        if self.cfg.start_date:
            df = df[df["published_at_utc"].dt.date >= self.cfg.start_date]
        if self.cfg.end_date:
            df = df[df["published_at_utc"].dt.date <= self.cfg.end_date]

        # De-duplicate (same ticker+title_norm)
        df["title_norm"] = df["title"].str.lower().fillna("").apply(normalize_title)
        df = df.sort_values(["ticker", "published_at_utc"]).drop_duplicates(
            subset=["ticker", "title_norm"], keep="first"
        )
        df["date_only"] = df["published_at_utc"].dt.date
        return df

    # ------------- Sentiment -------------
    def calculate_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logging.warning("No news to score.")
            return df
        texts = df["title"].fillna("").tolist()
        scores = self.scorer.score(texts)
        if not scores or len(scores) != len(df):
            logging.warning("Sentiment scoring failed or mismatched; filling zeros.")
            df["pos"] = 0.0
            df["neg"] = 0.0
            df["neu"] = 0.0
            df["compound"] = 0.0
            return df
        df = df.copy()
        df["pos"] = [s["pos"] for s in scores]
        df["neg"] = [s["neg"] for s in scores]
        df["neu"] = [s["neu"] for s in scores]
        df["compound"] = [s["compound"] for s in scores]
        return df

    # ------------- Aggregation -------------
    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Recency weighting (exponential decay with half-life)
        # w = exp(-lambda * age_hours), where lambda = ln(2) / half_life
        latest = df["published_at_utc"].max()
        age_hours = (latest - df["published_at_utc"]).dt.total_seconds() / 3600.0
        lam = math.log(2.0) / max(1e-9, float(self.cfg.half_life_hours))
        df["weight"] = np.exp(-lam * age_hours)

        # Daily (unweighted and weighted)
        agg = (df.groupby(["ticker", "date_only"])
                 .agg(avg_compound=("compound", "mean"),
                      avg_pos=("pos", "mean"),
                      avg_neg=("neg", "mean"),
                      avg_neu=("neu", "mean"),
                      news_count=("title", "count"),
                      wsum=("weight", "sum"),
                      wc=("compound", lambda s: np.average(s, weights=df.loc[s.index, "weight"]) if s.size else np.nan),
                      all_titles=("title", lambda s: " ".join(s.astype(str).tolist())))
                 .reset_index())

        agg = agg.rename(columns={"wc": "avg_compound_w", "date_only": "date"})
        agg["date"] = pd.to_datetime(agg["date"])
        agg = agg.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Rolling z-score on weighted compound to minimize look-ahead
        def rolling_z(s: pd.Series) -> pd.Series:
            mu = s.rolling(self.cfg.rolling_window, min_periods=max(5, self.cfg.rolling_window // 4)).mean()
            sd = s.rolling(self.cfg.rolling_window, min_periods=max(5, self.cfg.rolling_window // 4)).std()
            return (s - mu) / sd.replace({0.0: np.nan})

        agg["normalized_score"] = agg.groupby("ticker")["avg_compound_w"].apply(rolling_z).values
        agg["sentiment_change"] = agg.groupby("ticker")["avg_compound_w"].diff()
        return agg

    # ------------- Plots -------------
    def plot_ticker(self, agg: pd.DataFrame, ticker: str) -> None:
        outdir = os.path.join(self.cfg.base_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        df = agg[agg["ticker"] == ticker].copy()
        if df.empty:
            logging.warning("No data for plots: %s", ticker)
            return

        # Use seaborn if available
        use_sns = "seaborn" in sys.modules

        # Normalized score over time
        plt.figure(figsize=(12, 6))
        if use_sns:
            sns.lineplot(data=df, x="date", y="normalized_score", marker="o")
        else:
            plt.plot(df["date"], df["normalized_score"], marker="o")
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.title(f"Rolling-Normalized Sentiment (weighted) — {ticker}")
        plt.xlabel("Date"); plt.ylabel("Z-Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{ticker}_sentiment_norm.png"), dpi=300)
        plt.close()

        # News Volume
        plt.figure(figsize=(12, 6))
        if use_sns:
            sns.barplot(data=df, x="date", y="news_count")
        else:
            plt.bar(df["date"].dt.strftime("%Y-%m-%d"), df["news_count"])
        plt.title(f"News Volume — {ticker}")
        plt.xlabel("Date"); plt.ylabel("Articles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{ticker}_news_volume.png"), dpi=300)
        plt.close()

        # Word cloud (if library available)
        if "WordCloud" in globals() and df["all_titles"].notna().any():
            text = " ".join([x for x in df["all_titles"].astype(str).tolist() if x.strip()])
            if text.strip():
                try:
                    wc = WordCloud(width=1000, height=500, background_color="white", max_words=150).generate(text)
                    plt.figure(figsize=(12, 7))
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    plt.title(f"Word Cloud — {ticker}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir, f"{ticker}_wordcloud.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    logging.warning("WordCloud failed for %s: %s", ticker, e)

    def plot_comparative(self, agg: pd.DataFrame) -> None:
        outdir = os.path.join(self.cfg.base_dir, "plots")
        os.makedirs(outdir, exist_ok=True)
        if agg.empty:
            return
        # Last up-to-10 dates
        dates = sorted(agg["date"].unique())[-min(10, agg["date"].nunique()):]
        comp = agg[agg["date"].isin(dates)]
        if comp.empty:
            return

        use_sns = "seaborn" in sys.modules

        # Comparative sentiment (weighted avg)
        plt.figure(figsize=(14, 7))
        if use_sns:
            sns.lineplot(data=comp, x="date", y="avg_compound_w", hue="ticker", marker="o")
        else:
            for t, g in comp.groupby("ticker"):
                plt.plot(g["date"], g["avg_compound_w"], marker="o", label=t)
            plt.legend()
        plt.title("Comparative Sentiment (weighted average)")
        plt.xlabel("Date"); plt.ylabel("Avg Compound (weighted)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "comparative_sentiment.png"), dpi=300)
        plt.close()

        # News volume by ticker/date
        plt.figure(figsize=(14, 7))
        if use_sns:
            sns.barplot(data=comp, x="ticker", y="news_count", hue="date")
        else:
            # simple grouped bars (fallback)
            tickers = list(sorted(comp["ticker"].unique()))
            idx = np.arange(len(tickers))
            widths = 0.8 / max(1, len(dates))
            for i, d in enumerate(dates):
                vals = [int(comp[(comp["ticker"] == t) & (comp["date"] == d)]["news_count"].sum()) for t in tickers]
                plt.bar(idx + i*widths, vals, width=widths, label=str(pd.to_datetime(d).date()))
            plt.xticks(idx + widths*(len(dates)-1)/2, tickers)
            plt.legend()
        plt.title("News Volume by Ticker & Date")
        plt.xlabel("Ticker"); plt.ylabel("Articles")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "comparative_volume.png"), dpi=300)
        plt.close()

        # Heatmap (weighted compound) if seaborn available
        if use_sns and comp["ticker"].nunique() > 1 and comp["date"].nunique() > 1:
            pivot = comp.pivot(index="ticker", columns="date", values="avg_compound_w")
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, cmap="RdYlGn", center=0)
            plt.title("Sentiment Heatmap (weighted compound)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "sentiment_heatmap.png"), dpi=300)
            plt.close()

    # ------------- Export -------------
    def export(self, news_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(self.cfg.base_dir, "results")
        os.makedirs(outdir, exist_ok=True)
        news_df.to_csv(os.path.join(outdir, f"news_data_{ts}.csv"), index=False)
        daily_df.to_csv(os.path.join(outdir, f"daily_sentiment_{ts}.csv"), index=False)
        logging.info("Exported CSVs to %s", outdir)

    # ------------- Diagnostics (optional) -------------
    def diagnostics(self, daily_df: pd.DataFrame) -> None:
        """
        Basic utility: correlate today's sentiment with next-day returns;
        simple long/short on top/bottom quintiles (requires yfinance).
        """
        if "yfinance" not in sys.modules:
            logging.info("yfinance not available; skipping diagnostics/backtest.")
            return
        if daily_df.empty:
            return

        logging.info("Running basic diagnostics/backtest (requires network).")

        # --- Per-ticker next-day correlation ---
        diags = []
        for t in sorted(daily_df["ticker"].unique()):
            g = daily_df[daily_df["ticker"] == t].sort_values("date").copy()
            start = (g["date"].min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            end = (g["date"].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

            try:
                # Single-ticker history -> single-index columns
                hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            except Exception as e:
                logging.warning("yfinance history() failed for %s: %s", t, e)
                continue
            if hist.empty:
                continue

            px = hist.reset_index().rename(columns={"Date": "date"})
            # Ensure naive datetime (no tz) for merge
            px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
            g["date"] = pd.to_datetime(g["date"]).dt.tz_localize(None)

            # Align by nearest date (asof) within 1 day tolerance
            px = px.sort_values("date")[["date", "Close"]]
            g = g.sort_values("date")

            merged = pd.merge_asof(
                g,
                px,
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta("1D")
            )

            # Compute next-day return from prices
            merged["next_close"] = merged["Close"].shift(-1)
            merged["next_ret"] = (merged["next_close"] / merged["Close"] - 1.0)
            merged = merged.dropna(subset=["next_ret", "avg_compound_w"])

            if merged.empty:
                continue

            pear = merged["avg_compound_w"].corr(merged["next_ret"])
            spear = merged["avg_compound_w"].corr(merged["next_ret"], method="spearman")
            diags.append((t, float(pear) if pear == pear else np.nan, float(spear) if spear == spear else np.nan))

        if diags:
            logging.info("Per-ticker correlation (avg_compound_w vs next-day return):")
            for t, p, s in diags:
                logging.info("  %s: pearson=%.4f, spearman=%.4f", t, p, s)

        # --- Cross-sectional long/short: top vs bottom 20% normalized_score ---
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        uniq = sorted(df["ticker"].unique())
        try:
            # Multi-ticker: handle both single-index and MultiIndex returns
            px_all = yf.download(
                uniq,
                start=(df["date"].min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                end=(df["date"].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                group_by="ticker"
            )
        except Exception as e:
            logging.warning("yfinance multi-ticker download failed: %s", e)
            return

        # Build tidy prices table
        rets_frames = []
        if isinstance(px_all.columns, pd.MultiIndex):
            for t in uniq:
                if t in px_all.columns.get_level_values(0):
                    sub = px_all[t]
                    if "Close" in sub.columns:
                        tmp = sub["Close"].dropna().rename("Close").reset_index()
                        tmp["ticker"] = t
                        tmp.rename(columns={"Date": "date"}, inplace=True)
                        rets_frames.append(tmp)
        else:
            # Single ticker case or flattened structure
            if "Close" in px_all.columns:
                tmp = px_all["Close"].dropna().rename("Close").reset_index()
                tmp["ticker"] = uniq[0]
                tmp.rename(columns={"Date": "date"}, inplace=True)
                rets_frames.append(tmp)

        if not rets_frames:
            logging.info("No price data available for backtest.")
            return

        prices = pd.concat(rets_frames, ignore_index=True)
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
        prices = prices.sort_values(["ticker", "date"])
        prices["ret_next"] = prices.groupby("ticker")["Close"].pct_change().shift(-1)

        merged = pd.merge(
            df,
            prices[["ticker", "date", "ret_next"]],
            on=["ticker", "date"],
            how="left"
        ).dropna(subset=["ret_next", "normalized_score"])

        daily_pnl = []
        for d, g in merged.groupby("date"):
            g = g.sort_values("normalized_score")
            n = len(g)
            if n < 5:
                continue
            k = max(1, n // 5)  # 20%
            short = g.iloc[:k]["ret_next"].mean()
            long = g.iloc[-k:]["ret_next"].mean()
            daily_pnl.append((d, long - short))

        if daily_pnl:
            pnl_df = pd.DataFrame(daily_pnl, columns=["date", "pnl"])
            sharpe = pnl_df["pnl"].mean() / (pnl_df["pnl"].std() + 1e-9) * math.sqrt(252)
            logging.info("Toy long/short (top vs bottom 20%%) Sharpe ~ %.2f, avg daily pnl %.4f%%",
                         sharpe, pnl_df["pnl"].mean() * 100.0)

    # ------------- Main pipeline -------------
    def analyze(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = self.fetch_all_news()
        news_df = self.process_news_data(raw)
        if news_df.empty:
            logging.warning("No news parsed for any ticker.")
            self.news_df = news_df
            self.daily_sentiment_df = pd.DataFrame()
            return news_df, self.daily_sentiment_df

        news_df = self.calculate_sentiment(news_df)
        daily = self.aggregate_daily(news_df)

        # Summary logging
        logging.info("News rows: %d | Daily rows: %d", len(news_df), len(daily))
        if not daily.empty:
            for t in sorted(daily["ticker"].unique()):
                sub = daily[daily["ticker"] == t]
                logging.info("%s: %d days, date range %s to %s, total articles=%d",
                             t, len(sub),
                             sub["date"].min().date(), sub["date"].max().date(),
                             int(news_df[news_df['ticker'] == t].shape[0]))

        self.news_df = news_df
        self.daily_sentiment_df = daily
        return news_df, daily

    def run(self):
        news_df, daily_df = self.analyze()

        if daily_df.empty:
            logging.warning("No daily sentiment available; exiting.")
            return

        # Plots
        if self.cfg.make_plots:
            for t in sorted(self.cfg.tickers):
                self.plot_ticker(daily_df, t)
            self.plot_comparative(daily_df)
            logging.info("Plots saved to %s", os.path.join(self.cfg.base_dir, "plots"))

        # Export
        if self.cfg.export_results:
            self.export(news_df, daily_df)

        # Diagnostics/backtest
        if self.cfg.backtest:
            self.diagnostics(daily_df)

        # Console summary (top headlines with scores)
        top = news_df.sort_values("published_at_utc", ascending=False).head(10)[
            ["ticker", "published_at_utc", "title", "compound"]
        ]
        logging.info("Recent headlines (top 10):")
        for _, r in top.iterrows():
            logging.info("[%s] %s | score=%.3f | %s", r["ticker"], r["published_at_utc"], r["compound"], r["title"])

        # Flag low-article days
        few = daily_df[daily_df["news_count"] < self.cfg.min_articles_per_day_flag]
        if not few.empty:
            logging.info("Days with < %d articles: %d", self.cfg.min_articles_per_day_flag, len(few))


# =========================
# CLI
# =========================

def parse_args(argv: Optional[List[str]] = None) -> AnalyzerConfig:
    ap = argparse.ArgumentParser(description="OSTA Stock News Sentiment (Finviz)")

    ap.add_argument("--tickers", nargs="+", required=True,
                    help="List of stock tickers, e.g., AAPL MSFT GOOGL")
    ap.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    ap.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR, help="Output base directory")
    ap.add_argument("--no-cache", action="store_true", help="Disable requests caching")
    ap.add_argument("--cache-ttl", type=int, default=1800, help="Cache TTL seconds (default 1800)")
    ap.add_argument("--sentiment-model", type=str, default="finbert", choices=["finbert", "vader"],
                    help="Sentiment model to use")
    ap.add_argument("--half-life", type=float, default=24.0, help="Half-life hours for recency weighting")
    ap.add_argument("--rolling-window", type=int, default=20, help="Rolling window for z-score")
    ap.add_argument("--export", action="store_true", help="Export CSVs")
    ap.add_argument("--no-export", action="store_true", help="Disable CSV export")
    ap.add_argument("--plots", action="store_true", help="Generate plots")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots")
    ap.add_argument("--backtest", action="store_true", help="Run basic diagnostics/backtest (needs yfinance)")
    ap.add_argument("--verbose", action="store_true", help="More logging")

    args = ap.parse_args(argv)

    start_d = pd.to_datetime(args.start).date() if args.start else None
    end_d = pd.to_datetime(args.end).date() if args.end else None

    return AnalyzerConfig(
        tickers=args.tickers,
        base_dir=args.base_dir,
        use_cache=(not args.no_cache),
        cache_ttl_seconds=args.cache_ttl,
        sentiment_model=args.sentiment_model,
        half_life_hours=args.half_life,
        rolling_window=args.rolling_window,
        export_results=(args.export and not args.no_export),
        make_plots=(args.plots and not args.no_plots),
        backtest=args.backtest,
        start_date=start_d,
        end_date=end_d,
    )


def main():
    cfg = parse_args()
    setup_logging(cfg.base_dir, verbose=True)
    logging.info("Config: %s", cfg)

    # Gentle reminder about dependencies
    if cfg.sentiment_model.lower() == "finbert" and not FINBERT_AVAILABLE:
        logging.warning("FinBERT requested but transformers/torch are not installed. Using VADER instead.")

    analyzer = StockSentimentAnalyzer(cfg)
    analyzer.run()
    logging.info("Done.")


if __name__ == "__main__":
    main()
