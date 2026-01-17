"""
Pathway streaming tables for LiveBrief.
Defines core streaming tables for articles, events, and impact analysis.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from loguru import logger

from config.settings import settings


def utcnow() -> datetime:
    """Get current UTC time with timezone info"""
    return datetime.now(timezone.utc)


def _ensure_tz_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware"""
    if dt is None:
        return utcnow()
    
    if not hasattr(dt, 'tzinfo') or dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    
    return dt


class PathwayStreamingTables:
    """
    Manages Pathway-compatible streaming tables for LiveBrief.
    
    Tables:
    - articles_table: Raw articles from NewsData.io
    - events_table: Clustered news events
    - impact_table: Computed impact analysis
    - timeline_table: Event evolution timeline
    """
    
    def __init__(self):
        self._initialized = False
        self._articles_data: List[Dict] = []
        self._events_data: Dict[str, Dict] = {}
        self._impact_data: Dict[str, Dict] = {}
        self._timeline_data: List[Dict] = []
        self._clustering_state: Dict[str, Dict] = {}
        self._next_event_id = 0
    
    def initialize(self):
        """Initialize Pathway tables"""
        if not self._initialized:
            self._initialized = True
            logger.info("Pathway streaming tables initialized")
    
    def get_articles_table(self) -> List[Dict]:
        """Get the articles table"""
        if not self._initialized:
            self.initialize()
        return self._articles_data
    
    def get_events_table(self) -> Dict[str, Dict]:
        """Get the events table"""
        if not self._initialized:
            self.initialize()
        return self._events_data
    
    def get_impact_table(self) -> Dict[str, Dict]:
        """Get the impact table"""
        if not self._initialized:
            self.initialize()
        return self._impact_data
    
    def get_timeline_table(self) -> List[Dict]:
        """Get the timeline table"""
        if not self._initialized:
            self.initialize()
        return self._timeline_data
    
    def get_all_tables(self) -> Dict[str, Any]:
        """Get all streaming tables"""
        if not self._initialized:
            self.initialize()
        
        return {
            "articles": self.get_articles_table(),
            "events": self.get_events_table(),
            "impact": self.get_impact_table(),
            "timeline": self.get_timeline_table()
        }
    
    def add_article(self, article_data: Dict[str, Any]) -> None:
        """Add an article to the articles table"""
        if not self._initialized:
            self.initialize()
        
        self._articles_data.append(article_data)
        self._process_clustering(article_data)
    
    def _process_clustering(self, article_data: Dict[str, Any]) -> str:
        """Process article through clustering and assign event_id"""
        article_id = article_data["article_id"]
        embedding = article_data.get("embedding", [])
        
        event_id = self._find_best_event(article_data, embedding)
        
        if event_id:
            self._update_event(event_id, article_data)
        else:
            event_id = self._create_event(article_data)
        
        article_data["event_id"] = event_id
        return event_id
    
    def _find_best_event(
        self, 
        article_data: Dict[str, Any], 
        embedding: List[float]
    ) -> Optional[str]:
        """Find best matching event for an article"""
        import numpy as np
        
        best_event_id = None
        best_similarity = 0.25
        
        article_embedding = np.array(embedding) if embedding else None
        article_time = _ensure_tz_aware(article_data.get("published_at", utcnow()))
        article_words = set(article_data.get("title", "").lower().split())
        
        for event_id, event_data in self._clustering_state.items():
            event_time = _ensure_tz_aware(event_data.get("last_updated_at", utcnow()))
            
            try:
                hours_diff = (event_time - article_time).total_seconds() / 3600
            except (TypeError, AttributeError):
                continue
            
            if hours_diff > settings.pathway.event_window_hours:
                continue
            
            # Keyword overlap in titles
            event_words = set(event_data.get("title", "").lower().split())
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for", "and", "or", "but", "with", "by", "from", "as", "be", "have", "has", "had", "it", "this", "that"}
            article_key_words = article_words - stopwords
            event_key_words = event_words - stopwords
            
            if article_key_words and event_key_words:
                overlap = len(article_key_words & event_key_words)
                if overlap >= 1:
                    keyword_similarity = min(overlap / max(len(article_key_words), len(event_key_words)), 1.0)
                    if keyword_similarity > best_similarity:
                        best_similarity = keyword_similarity
                        best_event_id = event_id
                        continue
            
            # Embedding similarity
            if article_embedding is not None and event_data.get("embeddings"):
                center = np.mean(event_data["embeddings"], axis=0)
                if np.linalg.norm(center) > 0:
                    similarity = np.dot(article_embedding, center) / (
                        np.linalg.norm(article_embedding) * np.linalg.norm(center) + 1e-8
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_event_id = event_id
        
        return best_event_id
    
    def _create_event(self, article_data: Dict[str, Any]) -> str:
        """Create a new event from an article"""
        self._next_event_id += 1
        event_id = f"evt_{self._next_event_id:06d}"
        
        self._clustering_state[event_id] = {
            "title": article_data.get("title", ""),
            "description": article_data.get("description", ""),
            "article_ids": [article_data["article_id"]],
            "embeddings": [article_data.get("embedding", [])],
            "first_article_at": article_data.get("published_at", utcnow()),
            "last_updated_at": utcnow(),
            "categories": [article_data.get("category", "general")],
            "sources": [article_data.get("source", "Unknown")],
            "article_count": 1
        }
        
        logger.debug(f"Created new event {event_id}")
        return event_id
    
    def _update_event(self, event_id: str, article_data: Dict[str, Any]) -> None:
        """Update an existing event with new article"""
        if event_id not in self._clustering_state:
            return
        
        event = self._clustering_state[event_id]
        event["article_ids"].append(article_data["article_id"])
        event["embeddings"].append(article_data.get("embedding", []))
        event["last_updated_at"] = utcnow()
        event["article_count"] += 1
        
        category = article_data.get("category", "general")
        if category not in event["categories"]:
            event["categories"].append(category)
        
        source = article_data.get("source", "Unknown")
        if source not in event["sources"]:
            event["sources"].append(source)
        
        logger.debug(f"Updated event {event_id}")
    
    def get_event(self, event_id: str) -> Optional[Dict]:
        """Get event data by ID"""
        return self._clustering_state.get(event_id)
    
    def get_all_events(self) -> Dict[str, Dict]:
        """Get all events"""
        return self._clustering_state
    
    def cleanup_old_events(self, max_age_hours: int = None) -> int:
        """Remove events that haven't been updated recently"""
        if max_age_hours is None:
            max_age_hours = settings.pathway.event_window_hours
        
        cutoff = utcnow() - timedelta(hours=max_age_hours)
        removed = 0
        
        to_remove = []
        for eid, event in self._clustering_state.items():
            event_time = _ensure_tz_aware(event.get("last_updated_at", utcnow()))
            try:
                if event_time < cutoff:
                    to_remove.append(eid)
            except (TypeError, AttributeError):
                to_remove.append(eid)
        
        for event_id in to_remove:
            del self._clustering_state[event_id]
            removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old events")
        
        return removed
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about clustering state"""
        total_articles = sum(e["article_count"] for e in self._clustering_state.values())
        
        return {
            "total_events": len(self._clustering_state),
            "total_clustered_articles": total_articles,
            "avg_articles_per_event": total_articles / len(self._clustering_state) if self._clustering_state else 0
        }
    
    def generate_impact_for_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Generate impact analysis for an event"""
        event = self._clustering_state.get(event_id)
        if not event:
            return None
        
        try:
            from services.impact_generator import impact_generator
            from models.schemas import NewsEvent
            
            news_event = NewsEvent(
                event_id=event_id,
                title=event.get("title", ""),
                description=event.get("description", ""),
                article_ids=event.get("article_ids", []),
                first_article_at=event.get("first_article_at", utcnow()),
                last_updated_at=event.get("last_updated_at", utcnow()),
                categories=event.get("categories", []),
                sources=event.get("sources", []),
                article_count=event.get("article_count", 1)
            )
            
            impact = impact_generator.generate_impact(news_event)
            
            impact_data = {
                "affected_groups": impact.affected_groups,
                "daily_life_impact": impact.daily_life_impact,
                "short_term_risk": impact.short_term_risk,
                "what_to_know": impact.what_to_know,
                "generated_at": str(impact.generated_at),
                "confidence": impact.confidence
            }
            
            self._clustering_state[event_id]["impact"] = impact_data
            self._clustering_state[event_id]["impact_generated"] = True
            self._clustering_state[event_id]["impact_summary"] = impact.daily_life_impact
            self._clustering_state[event_id]["affected_groups"] = impact.affected_groups
            
            logger.info(f"Generated impact for event {event_id}")
            return impact_data
            
        except Exception as e:
            logger.error(f"Error generating impact for event {event_id}: {e}")
            return None


# Global streaming tables instance
pathway_tables = PathwayStreamingTables()
streaming_tables = pathway_tables


class EventClusteringState:
    """
    Maintains state for event clustering.
    Tracks article-event assignments and cluster metadata.
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self._event_clusters: Dict[str, Dict] = {}
        self._article_events: Dict[str, str] = {}
        self._next_event_id = 0
    
    def get_next_event_id(self) -> str:
        """Generate next unique event ID"""
        self._next_event_id += 1
        return f"evt_{self._next_event_id:06d}"
    
    def create_event(
        self, 
        article: Any, 
        embedding: List[float],
        title: str,
        description: str
    ) -> str:
        """Create a new event from an article"""
        event_id = self.get_next_event_id()
        
        self._event_clusters[event_id] = {
            "title": title,
            "description": description,
            "article_ids": [article.article_id] if hasattr(article, 'article_id') else [],
            "embeddings": [embedding],
            "first_article_at": article.published_at if hasattr(article, 'published_at') else utcnow(),
            "last_updated_at": article.published_at if hasattr(article, 'published_at') else utcnow(),
            "categories": [article.category.value] if hasattr(article, 'category') and hasattr(article.category, 'value') else ["general"],
            "sources": [article.source] if hasattr(article, 'source') else ["Unknown"],
            "sentiment": article.sentiment.value if hasattr(article, 'sentiment') and article.sentiment else "neutral",
            "article_count": 1
        }
        
        if hasattr(article, 'article_id'):
            self._article_events[article.article_id] = event_id
        
        logger.info(f"Created new event {event_id}")
        return event_id
    
    def add_to_event(
        self, 
        event_id: str, 
        article: Any, 
        embedding: List[float]
    ):
        """Add an article to an existing event"""
        if event_id not in self._event_clusters:
            logger.warning(f"Event {event_id} not found")
            return
        
        cluster = self._event_clusters[event_id]
        cluster["embeddings"].append(embedding)
        cluster["last_updated_at"] = utcnow()
        cluster["article_count"] += 1
        
        if hasattr(article, 'category') and hasattr(article.category, 'value'):
            if article.category.value not in cluster["categories"]:
                cluster["categories"].append(article.category.value)
        
        if hasattr(article, 'source'):
            if article.source not in cluster["sources"]:
                cluster["sources"].append(article.source)
        
        if hasattr(article, 'article_id'):
            cluster["article_ids"].append(article.article_id)
            self._article_events[article.article_id] = event_id
        
        logger.debug(f"Added article to event {event_id}")
    
    def assign_article(
        self, 
        article: Any, 
        embedding: List[float]
    ) -> str:
        """Assign an article to an existing event or create a new one"""
        best_event_id = None
        best_similarity = self.similarity_threshold
        
        for event_id, cluster in self._event_clusters.items():
            similarity = self._compute_similarity(embedding, cluster["embeddings"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_event_id = event_id
        
        if best_event_id:
            self.add_to_event(best_event_id, article, embedding)
            return best_event_id
        else:
            title = article.title if hasattr(article, 'title') else ""
            desc = article.description if hasattr(article, 'description') else ""
            return self.create_event(article, embedding, title=title, description=desc)
    
    def _compute_similarity(
        self, 
        embedding: List[float], 
        cluster_embeddings: List[List[float]]
    ) -> float:
        """Compute similarity between embedding and cluster center"""
        if not cluster_embeddings or not embedding:
            return 0.0
        
        import numpy as np
        
        embedding_arr = np.array(embedding)
        center_arr = np.mean(cluster_embeddings, axis=0)
        
        dot = np.dot(embedding_arr, center_arr)
        norm = np.linalg.norm(embedding_arr) * np.linalg.norm(center_arr)
        
        if norm == 0:
            return 0.0
        
        return dot / norm
    
    def get_event(self, event_id: str) -> Optional[Dict]:
        """Get event data by ID"""
        return self._event_clusters.get(event_id)
    
    def get_all_events(self) -> Dict[str, Dict]:
        """Get all events"""
        return self._event_clusters
    
    def cleanup_old_events(self, max_age_hours: int = 24) -> int:
        """Remove events that haven't been updated recently"""
        cutoff = utcnow() - timedelta(hours=max_age_hours)
        removed = 0
        
        to_remove = []
        for eid, cluster in self._event_clusters.items():
            event_time = _ensure_tz_aware(cluster.get("last_updated_at", utcnow()))
            try:
                if event_time < cutoff:
                    to_remove.append(eid)
            except (TypeError, AttributeError):
                to_remove.append(eid)
        
        for event_id in to_remove:
            del self._event_clusters[event_id]
            removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old events")
        
        return removed


# Global clustering state
clustering_state = EventClusteringState(
    similarity_threshold=settings.pathway.similarity_threshold
)

