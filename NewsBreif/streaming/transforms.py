"""
Streaming transformations for LiveBrief.
Implements incremental transformations for event clustering, feature engineering,
and real-time analysis.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import hashlib
from loguru import logger

from models.schemas import Article, NewsEvent, ImpactAnalysis, EventUpdate
from streaming.tables import pathway_tables, clustering_state
from config.settings import settings


def utcnow() -> datetime:
    """Get current UTC time with timezone info (timezone-aware)"""
    return datetime.now(timezone.utc)


class StreamingTransforms:
    """
    Implements streaming transformations for LiveBrief.
    
    Transformations:
    - Article embedding generation
    - Event clustering and assignment
    - Feature engineering (counts, sentiment, etc.)
    - Impact analysis triggering
    - Timeline updates
    """
    
    def __init__(self):
        self.similarity_threshold = settings.pathway.similarity_threshold
        self.event_window_hours = settings.pathway.event_window_hours
        self.batch_size = settings.pathway.batch_size
    
    def process_article(self, article_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single article and return enriched data.
        This is called when new articles arrive.
        """
        try:
            article_id = article_row["article_id"]
            
            # Generate embedding
            embedding = self._generate_embedding(
                article_row["title"], 
                article_row.get("description", "")
            )
            
            # Assign to event using pathway_tables
            event_id = pathway_tables._process_clustering({
                "article_id": article_id,
                "title": article_row["title"],
                "description": article_row.get("description", ""),
                "published_at": article_row.get("published_at", utcnow()),
                "category": article_row.get("category", "general"),
                "source": article_row.get("source", "Unknown"),
                "embedding": embedding
            })
            
            # Get updated event data
            event_data = pathway_tables.get_event(event_id)
            
            if not event_data:
                logger.error(f"No event data found for {event_id}")
                return article_row
            
            # Build result
            result = article_row.copy()
            result["event_id"] = event_id
            result["embedding"] = embedding
            
            # Mark if impact should be generated
            result["needs_impact"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return article_row
    
    def _generate_embedding(self, title: str, description: str) -> List[float]:
        """Generate embedding for article"""
        import numpy as np
        
        text = f"{title} {description}".lower()
        words = text.split()
        embedding = np.zeros(384)
        
        for i, word in enumerate(words[:100]):
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % 384
            embedding[idx] += 1
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def compute_event_features(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute features for each event from articles"""
        event_features = {}
        
        for article in articles:
            event_id = article.get("event_id")
            if event_id:
                if event_id not in event_features:
                    event_features[event_id] = {
                        "event_id": event_id,
                        "count": 0,
                        "categories": set(),
                        "sources": set(),
                        "first_article_at": None,
                        "last_updated_at": None
                    }
                
                features = event_features[event_id]
                features["count"] += 1
                features["categories"].add(article.get("category", "general"))
                features["sources"].add(article.get("source", "Unknown"))
        
        return event_features
    
    def compute_sentiment_trend(self, articles: List[Dict[str, Any]]) -> Dict[str, str]:
        """Compute sentiment trend for events"""
        event_sentiments: Dict[str, Dict[str, int]] = {}
        
        for article in articles:
            event_id = article.get("event_id")
            sentiment = article.get("sentiment", "neutral")
            
            if event_id:
                if event_id not in event_sentiments:
                    event_sentiments[event_id] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
                s = event_sentiments[event_id]
                if sentiment in s:
                    s[sentiment] += 1
                s["total"] += 1
        
        trends = {}
        for event_id, stats in event_sentiments.items():
            total = stats["total"] or 1
            pos_ratio = stats["positive"] / total
            neg_ratio = stats["negative"] / total
            if pos_ratio > neg_ratio + 0.2:
                trends[event_id] = "positive"
            elif neg_ratio > pos_ratio + 0.2:
                trends[event_id] = "negative"
            else:
                trends[event_id] = "neutral"
        
        return trends
    
    def detect_update_type(self, article: Dict[str, Any], existing_articles: List[Dict[str, Any]]) -> str:
        """Determine what type of update this article represents"""
        if not existing_articles:
            return "new"
        
        correction_keywords = ["correction", "retraction", "update", "amended"]
        title_lower = article.get("title", "").lower()
        
        for keyword in correction_keywords:
            if keyword in title_lower:
                return "correction"
        
        return "update"
    
    def build_timeline_update(self, article: Dict[str, Any], event_id: str, update_type: str) -> Dict[str, Any]:
        """Build a timeline update record from an article"""
        return {
            "update_id": f"{event_id}_{article['article_id']}",
            "event_id": event_id,
            "article_id": article["article_id"],
            "update_type": update_type,
            "summary": article.get("description", "")[:200],
            "published_at": article["published_at"],
            "is_major": update_type in ["correction", "new_angle"],
            "_time": utcnow()
        }


# Global transforms instance
transforms = StreamingTransforms()


class EventEvolutionTracker:
    """Tracks how events evolve over time"""
    
    def __init__(self):
        self._event_history: Dict[str, List[Dict]] = {}
        self._major_update_threshold = 3
    
    def record_article(self, event_id: str, article: Dict[str, Any]):
        """Record an article for an event"""
        if event_id not in self._event_history:
            self._event_history[event_id] = []
        
        self._event_history[event_id].append({
            "article_id": article["article_id"],
            "published_at": article["published_at"],
            "title": article["title"],
            "update_type": article.get("update_type", "new")
        })
        
        # Keep history bounded
        if len(self._event_history[event_id]) > 100:
            self._event_history[event_id] = self._event_history[event_id][-50:]
    
    def should_regenerate_impact(self, event_id: str) -> bool:
        """Determine if impact analysis should be regenerated"""
        history = self._event_history.get(event_id, [])
        if not history:
            return True
        
        recent = []
        now = utcnow()
        for a in history:
            try:
                published_at = datetime.fromisoformat(a["published_at"].replace("Z", "+00:00"))
                # Ensure timezone awareness
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                if published_at > now - timedelta(hours=1):
                    recent.append(a)
            except (ValueError, TypeError, AttributeError):
                # If parsing fails, skip this article
                continue
        
        return len(recent) >= self._major_update_threshold
    
    def get_event_evolution(self, event_id: str) -> List[Dict]:
        """Get the evolution timeline for an event"""
        return self._event_history.get(event_id, [])


# Global evolution tracker
evolution_tracker = EventEvolutionTracker()
