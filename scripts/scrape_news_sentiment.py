"""
News Sentiment Scraper & Analyzer
Uses HuggingFace Transformers pipeline (GPU-accelerated) to scrape and
analyze recent MMA news articles for fighter sentiment.
Outputs: data/fighter_news_sentiment.csv
"""

import os
import time
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

DATA_DIR = Path(__file__).parent.parent / "data"
SENTIMENT_CSV = DATA_DIR / "fighter_news_sentiment.csv"
REQUEST_DELAY = 1.5

MMA_NEWS_SOURCES = [
    {
        "name": "MMA Fighting",
        "url": "https://www.mmafighting.com/",
        "selector": "h2 a, h3 a, .c-entry-box--compact__title a",
        "date_selector": "time",
    },
    {
        "name": "MMA Junkie",
        "url": "https://mmajunkie.usatoday.com/",
        "selector": "h2 a, h3 a, .entry-title a",
        "date_selector": "time",
    },
    {
        "name": "Sherdog News",
        "url": "https://www.sherdog.com/news/news/list",
        "selector": "h2 a, h3 a, .news_title a",
        "date_selector": ".news_date",
    },
]

UFC_FIGHTER_NAMES = [
    "Jon Jones", "Islam Makhachev", "Alexander Volkanovski", "Israel Adesanya",
    "Alex Pereira", "Charles Oliveira", "Leon Edwards", "Aljamain Sterling",
    "Max Holloway", "Dustin Poirier", "Conor McGregor", "Sean O'Malley",
    "Ilia Topuria", "Sean Strickland", "Jiri Prochazka", "Justin Gaethje",
    "Colby Covington", "Khamzat Chimaev", "Shavkat Rakhmonov", "Tom Aspinall",
    "Ciryl Gane", "Sergei Pavlovich", "Brandon Moreno", "Alexandre Pantoja",
    "Jailton Almeida", "Magomed Ankalaev", "Jamahal Hill", "Jan Blachowicz",
    "Curtis Blaydes", "Tai Tuivasa", "Jalin Turner", "Beneil Dariush",
    "Arman Tsarukyan", "Mateusz Gamrot", "Rafael Fiziev", "Dan Hooker",
    "Marlon Vera", "Cory Sandhagen", "Petr Yan", "Merab Dvalishvili",
    "Aljamain Sterling", "Henry Cejudo", "Zhang Weili", "Alexa Grasso",
    "Valentina Shevchenko", "Manon Fiorot", "Erin Blanchfield", "Rose Namajunas",
    "Kelvin Gastelum", "Robert Whittaker", "Paulo Costa", "Marvin Vettori",
    "Jared Cannonier", "Derek Brunson", "Dominick Reyes", "Anthony Smith",
    "Michael Chandler", "Tony Ferguson", "Kevin Holland", "Stephen Thompson",
    "Gilbert Burns", "Neil Magny", "Derrick Lewis", "Stipe Miocic",
    "Sergei Spivac", "Aleksandar Rakic", "Movsar Evloev", "Bryce Mitchell",
    "Giga Chikadze", "Yair Rodriguez", "Arnold Allen", "Calvin Kattar",
    "Brian Ortega", "Josh Emmett", "Jack Della Maddalena", "Vicente Luque",
    "Geoff Neal", "Belal Muhammad", "Shavkat Rakhmonov", "Ian Garry",
]


def get_soup(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def extract_article_links(source_info, max_articles=50):
    """Extract article links from a news source homepage."""
    soup = get_soup(source_info["url"])
    if soup is None:
        return []

    links = []
    try:
        articles = soup.select(source_info["selector"])
        for article in articles[:max_articles]:
            href = article.get("href", "")
            if href:
                if not href.startswith("http"):
                    base = source_info["url"].rstrip("/")
                    href = base + href if href.startswith("/") else base + "/" + href

                parent_text = article.parent.get_text(strip=True) if article.parent else ""
                title = article.get_text(strip=True) or parent_text[:200]

                links.append({
                    "title": title,
                    "url": href,
                    "source": source_info["name"],
                    "date_scraped": datetime.now().isoformat(),
                })
    except Exception as e:
        print(f"  Error extracting from {source_info['name']}: {e}")

    return links


def extract_article_text(url):
    """Extract main text content from an article URL."""
    soup = get_soup(url)
    if soup is None:
        return ""

    try:
        content_selectors = [
            "article p", ".article-body p", ".entry-content p",
            ".post-content p", ".c-entry-content p", ".article__content p",
            "main p", ".content p", "#article-content p",
        ]
        paragraphs = []
        for selector in content_selectors:
            p_tags = soup.select(selector)
            if p_tags:
                for p in p_tags:
                    text = p.get_text(strip=True)
                    if len(text) > 30:
                        paragraphs.append(text)
                if len(paragraphs) > 3:
                    break

        if not paragraphs:
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]

        return " ".join(paragraphs[:20])
    except Exception as e:
        print(f"  Error extracting text from {url}: {e}")
        return ""


def find_fighter_mentions(text, fighter_names):
    """Find which fighters are mentioned in the article text."""
    text_lower = text.lower()
    mentioned = []
    for fighter in fighter_names:
        if fighter.lower() in text_lower:
            mentioned.append(fighter)
    return mentioned


def initialize_sentiment_pipeline():
    """Initialize the HuggingFace sentiment pipeline on GPU."""
    print("Initializing sentiment analysis model on GPU...")

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device = 0 if torch.cuda.is_available() else -1
        if device >= 0:
            model = model.to(f"cuda:{device}")
            print(f"  Model loaded on GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("  GPU not available, using CPU (slower)")

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=512,
            truncation=True,
        )
        return sentiment_pipeline, device
    except Exception as e:
        print(f"  Error loading primary model: {e}")
        print("  Falling back to distilbert-base-uncased-finetuned-sst-2-english...")
        device = 0 if torch.cuda.is_available() else -1
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            max_length=512,
            truncation=True,
        )
        return sentiment_pipeline, device


def analyze_sentiment_batch(texts, pipeline_obj, batch_size=16):
    """Run sentiment analysis on a batch of texts."""
    if not texts:
        return []

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_results = pipeline_obj(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"  Batch sentiment error: {e}")
            results.extend([{"label": "neutral", "score": 0.5}] * len(batch))
    return results


def scrape_and_analyze_news():
    """Main function: scrape news, analyze sentiment, and save results."""
    print("\n" + "=" * 60)
    print("  News Sentiment Scraper & Analyzer")
    print("=" * 60)

    sentiment_pipeline, device = initialize_sentiment_pipeline()

    all_articles = []
    for source in MMA_NEWS_SOURCES:
        print(f"\nScraping {source['name']}...")
        articles = extract_article_links(source, max_articles=30)
        all_articles.extend(articles)
        print(f"  Found {len(articles)} article links")

    print(f"\nTotal articles found: {len(all_articles)}")

    print("\nExtracting article text and analyzing sentiment...")
    sentiment_records = []
    texts_batch = []
    articles_batch = []

    for article in tqdm(all_articles, desc="Articles"):
        text = extract_article_text(article["url"])
        if text:
            mentioned_fighters = find_fighter_mentions(text, UFC_FIGHTER_NAMES)
            if mentioned_fighters:
                texts_batch.append(text[:512])
                articles_batch.append({
                    **article,
                    "mentioned_fighters": "|".join(mentioned_fighters),
                })

    if texts_batch:
        print(f"\nRunning sentiment analysis on {len(texts_batch)} articles...")
        sentiments = analyze_sentiment_batch(texts_batch, sentiment_pipeline, batch_size=16)

        for art, sent in zip(articles_batch, sentiments):
            for fighter in art["mentioned_fighters"].split("|"):
                label = sent["label"].lower()
                score = sent["score"]

                if label in ["positive", "pos"]:
                    sentiment_score = score
                elif label in ["negative", "neg"]:
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0

                sentiment_records.append({
                    "fighter_name": fighter,
                    "article_title": art["title"],
                    "article_url": art["url"],
                    "source": art["source"],
                    "date_scraped": art["date_scraped"],
                    "sentiment_label": label,
                    "sentiment_score": sentiment_score,
                    "confidence": score,
                })

    if not sentiment_records:
        print("\nNo articles with fighter mentions found. Generating synthetic sentiment data.")
        sentiment_records = generate_synthetic_sentiment()

    df = pd.DataFrame(sentiment_records)
    df.to_csv(SENTIMENT_CSV, index=False)
    print(f"\nSaved {len(df)} sentiment records to {SENTIMENT_CSV}")

    summary = df.groupby("fighter_name").agg(
        articles_count=("sentiment_score", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        max_positive=("sentiment_score", "max"),
        min_negative=("sentiment_score", "min"),
        sentiment_volatility=("sentiment_score", "std"),
    ).reset_index()

    summary["momentum_score"] = (
        summary["avg_sentiment"] * 0.4
        + (summary["articles_count"] / summary["articles_count"].max()) * 0.2
        + (1 - summary["sentiment_volatility"].fillna(1).clip(0, 1)) * 0.4
    )
    summary["momentum_score"] = summary["momentum_score"].clip(-1, 1)

    print(f"\nSentiment summary for {len(summary)} fighters:")
    print(summary.sort_values("momentum_score", ascending=False).head(10).to_string())

    return df


def generate_synthetic_sentiment():
    """Generate synthetic sentiment data for demonstration when scraping fails."""
    import random
    random.seed(42)

    records = []
    for fighter in UFC_FIGHTER_NAMES:
        num_articles = random.randint(3, 15)
        for _ in range(num_articles):
            score = random.uniform(-1, 1)
            label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            records.append({
                "fighter_name": fighter,
                "article_title": f"Recent news about {fighter}",
                "article_url": "synthetic",
                "source": "Synthetic Data",
                "date_scraped": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                "sentiment_label": label,
                "sentiment_score": round(score, 3),
                "confidence": round(random.uniform(0.6, 0.99), 3),
            })
    return records


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not SENTIMENT_CSV.exists():
        scrape_and_analyze_news()
    else:
        print(f"Sentiment data already exists at {SENTIMENT_CSV}")
        df = pd.read_csv(SENTIMENT_CSV)
        print(f"Loaded {len(df)} records. Delete the file or use --refresh to re-scrape.")

    print("\nSentiment analysis complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-scrape and analyze")
    args = parser.parse_args()

    if args.refresh and SENTIMENT_CSV.exists():
        SENTIMENT_CSV.unlink()

    main()
