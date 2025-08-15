import aiohttp
import asyncio
import logging
from typing import Dict, Optional
from textblob import TextBlob

logger = logging.getLogger(__name__)

async def fetch_news(symbol: str, api_key: str) -> Optional[Dict]:
    """Fetch news articles for a given symbol using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data["status"] == "ok" and data["totalResults"] > 0:
                        return data["articles"]
                    else:
                        logger.warning(f"No articles found for {symbol}")
                        return None
                else:
                    logger.error(f"NewsAPI request failed with status {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return None

async def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of a given text using TextBlob."""
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # Polarity ranges from -1 to 1
        return sentiment
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return 0.0

async def get_realtime_news_and_sentiment(symbol: str, news_api_key: str) -> Optional[Dict]:
    """Fetch news and analyze sentiment in real-time."""
    news = await fetch_news(symbol, news_api_key)
    if news:
        sentiments = []
        for article in news:
            sentiment = await analyze_sentiment(article["title"] + " " + article["description"])
            sentiments.append(sentiment)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        return {"news": news, "avg_sentiment": avg_sentiment}
    else:
        return None
