"""
NewsData.io API Connector for LiveBrief.
Provides real-time news streaming via polling mechanism.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from loguru import logger

from config.settings import settings
from models.schemas import Article, ArticleCategory


def utcnow() -> datetime:
    """Get current UTC time with timezone info (timezone-aware)"""
    return datetime.now(timezone.utc)


class NewsDataConnector:
    """
    NewsData.io API connector for real-time news streaming.
    Implements polling mechanism for continuous data ingestion.
    """
    
    def __init__(self):
        self.api_key = settings.newsdata.api_key
        self.base_url = settings.newsdata.base_url
        self.language = settings.newsdata.language
        self.categories = settings.newsdata.categories
        self.country = settings.newsdata.country
        self.timeout = settings.newsdata.timeout
        self.polling_interval = settings.pathway.polling_interval
        
        self.client: Optional[httpx.Client] = None
        self._last_fetched_ids: set = set()
        self._running = False
        
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client"""
        if self.client is None:
            self.client = httpx.Client(timeout=self.timeout)
        return self.client
    
    def _close_client(self):
        """Close HTTP client"""
        if self.client:
            self.client.close()
            self.client = None
    
    def build_url(self, category: str = None) -> str:
        """Build NewsData API URL with parameters"""
        params = {
            "apikey": self.api_key,
            "language": self.language,
        }
        
        # Add country if specified
        if self.country:
            params["country"] = self.country
        
        # Add single category if specified
        if category:
            params["category"] = category
        
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.base_url}?{param_str}"
    
    def fetch_news_by_category(self, category: str) -> List[Article]:
        """Fetch news articles for a specific category"""
        try:
            client = self._get_client()
            url = self.build_url(category)
            
            logger.debug(f"Fetching news from NewsData ({category}): {url[:80]}...")
            
            response = client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "success":
                logger.error(f"NewsData API error for {category}: {data.get('results', 'Unknown error')}")
                return []
            
            articles_data = data.get("results", [])
            articles = []
            
            for article_data in articles_data:
                article = self.parse_article(article_data)
                if article:
                    articles.append(article)
            
            logger.debug(f"Fetched {len(articles)} articles for category: {category}")
            return articles
            
        except httpx.TimeoutException:
            logger.error(f"NewsData API request timed out for category: {category}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"NewsData API HTTP error for {category}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {category}: {e}")
            return []
    
    def fetch_news(self) -> List[Article]:
        """Fetch news articles from all configured categories"""
        import asyncio
        import httpx
        
        # Parse comma-separated categories
        categories_list = [c.strip() for c in self.categories.split(",") if c.strip()]
        
        if not categories_list:
            # No categories configured, fetch all news
            logger.info("No categories configured, fetching all news...")
            return self.fetch_news_by_category(None)
        
        async def fetch_all_categories():
            """Fetch all categories concurrently"""
            async with httpx.AsyncClient(timeout=60.0) as client:
                tasks = []
                for category in categories_list:
                    url = self.build_url(category)
                    tasks.append(self._fetch_async(client, url, category))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect all articles
                all_articles = []
                seen_ids = set()
                
                for result in results:
                    if isinstance(result, list):
                        for article in result:
                            if article.article_id not in seen_ids:
                                seen_ids.add(article.article_id)
                                all_articles.append(article)
                
                return all_articles
        
        # Run async fetch
        try:
            all_articles = asyncio.run(fetch_all_categories())
            logger.info(f"Fetched {len(all_articles)} total articles from {len(categories_list)} categories")
            return all_articles
        except Exception as e:
            logger.error(f"Error fetching all categories: {e}")
            # Fallback to single category
            return self.fetch_news_by_category(None)
    
    async def _fetch_async(self, client: httpx.AsyncClient, url: str, category: str) -> List[Article]:
        """Helper to fetch a single category asynchronously"""
        try:
            logger.debug(f"Fetching news from NewsData ({category}): {url[:80]}...")
            
            response = await client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "success":
                logger.error(f"NewsData API error for {category}: {data.get('results', 'Unknown error')}")
                return []
            
            articles_data = data.get("results", [])
            articles = []
            
            for article_data in articles_data:
                article = self.parse_article(article_data)
                if article:
                    articles.append(article)
            
            logger.debug(f"Fetched {len(articles)} articles for category: {category}")
            return articles
            
        except httpx.TimeoutException:
            logger.error(f"NewsData API request timed out for category: {category}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"NewsData API HTTP error for {category}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {category}: {e}")
            return []
    
    def fetch_new_articles(self) -> List[Article]:
        """Fetch only new articles since last fetch"""
        all_articles = self.fetch_news()
        
        # Filter to only new articles
        new_articles = [
            a for a in all_articles 
            if a.article_id not in self._last_fetched_ids
        ]
        
        # Update tracking set
        for article in all_articles:
            self._last_fetched_ids.add(article.article_id)
        
        # Keep set bounded
        if len(self._last_fetched_ids) > 10000:
            self._last_fetched_ids = set(list(self._last_fetched_ids)[-5000:])
        
        if new_articles:
            logger.info(f"Found {len(new_articles)} new articles")
        
        return new_articles
    
    def parse_article(self, article_data: Dict[str, Any]) -> Optional[Article]:
        """Parse NewsData API response into Article model"""
        try:
            # Handle published_at timestamp
            published_at_str = article_data.get("pubDate", "")
            if isinstance(published_at_str, str) and published_at_str:
                try:
                    published_at = datetime.fromisoformat(
                        published_at_str.replace("Z", "+00:00").replace("+00:00", "")
                    )
                    # Ensure timezone awareness
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    published_at = utcnow()
            else:
                published_at = utcnow()
            
            # Map category to enum
            category_raw = article_data.get("category", ["general"])[0] if article_data.get("category") else "general"
            category_map = {
                "technology": ArticleCategory.TECHNOLOGY,
                "business": ArticleCategory.BUSINESS,
                "health": ArticleCategory.HEALTH,
                "science": ArticleCategory.SCIENCE,
                "entertainment": ArticleCategory.ENTERTAINMENT,
                "sports": ArticleCategory.SPORTS,
                "general": ArticleCategory.GENERAL,
                "world": ArticleCategory.GENERAL,
                "nation": ArticleCategory.GENERAL,
                "economy": ArticleCategory.BUSINESS,
                "politics": ArticleCategory.GENERAL,
            }
            category = category_map.get(category_raw.lower(), ArticleCategory.GENERAL)
            
            # Extract author - API may return list like ["globe newswire"] or string
            author_raw = article_data.get("creator")
            if isinstance(author_raw, list) and author_raw:
                author = ", ".join([str(a) for a in author_raw if a]) or None
            elif isinstance(author_raw, str):
                author = author_raw if author_raw else None
            else:
                author = None
            
            # Extract country - API returns list like ["us"]
            country_raw = article_data.get("country", "us")
            if isinstance(country_raw, list) and country_raw:
                country = country_raw[0] if len(country_raw) > 0 else "us"
            elif isinstance(country_raw, str):
                country = country_raw
            else:
                country = "us"
            
            # Extract source name
            source_name = "Unknown"
            if article_data.get("source_id"):
                source_name = article_data.get("source_id")
            elif article_data.get("source"):
                if isinstance(article_data.get("source"), str):
                    source_name = article_data.get("source")
                elif isinstance(article_data.get("source"), dict):
                    source_name = article_data.get("source", {}).get("name", "Unknown")
            
            article = Article(
                article_id=article_data.get("article_id", article_data.get("link", "")),
                title=article_data.get("title", ""),
                description=article_data.get("description"),
                content=article_data.get("content"),
                source=source_name,
                author=author,
                url=article_data.get("link", ""),
                image_url=article_data.get("image_url"),
                category=category,
                language=article_data.get("language", "en"),
                country=country,
                published_at=published_at,
                fetched_at=utcnow()
            )
            
            return article
            
        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None
    
    def to_dict(self, article: Article) -> Dict[str, Any]:
        """Convert article to dictionary for processing"""
        return {
            "article_id": article.article_id,
            "title": article.title,
            "description": article.description,
            "content": article.content,
            "source": article.source,
            "author": article.author,
            "url": article.url,
            "image_url": article.image_url,
            "category": article.category.value,
            "language": article.language,
            "country": article.country,
            "published_at": article.published_at,
            "fetched_at": article.fetched_at
        }


def create_newsdata_connector() -> NewsDataConnector:
    """Factory function to create NewsData connector"""
    return NewsDataConnector()


def run_polling_loop(
    callback, 
    interval: int = None
):
    """
    Run polling loop with callback for each new article.
    
    Args:
        callback: Function to call for each new article
        interval: Polling interval in seconds
    """
    connector = create_newsdata_connector()
    if interval:
        connector.polling_interval = interval
    
    try:
        for article in connector.fetch_new_articles():
            callback(article)
    finally:
        connector._close_client()


if __name__ == "__main__":
    # Test the connector
    connector = create_newsdata_connector()
    articles = connector.fetch_new_articles()
    print(f"Fetched {len(articles)} articles")
    for article in articles[:3]:
        print(f"- {article.title[:80]}...")

