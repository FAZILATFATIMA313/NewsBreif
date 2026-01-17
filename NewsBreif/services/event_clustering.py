"""
Event Clustering Service for LiveBrief.
Clusters related articles using semantic similarity and keyword analysis.
Uses Pathway-compatible data structures for streaming.
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import hashlib
import re
from collections import defaultdict
from loguru import logger

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import Article, EventCluster
from config.settings import settings
from streaming.tables import pathway_tables


def utcnow() -> datetime:
    """Get current UTC time with timezone info (timezone-aware)"""
    return datetime.now(timezone.utc)


class EventClusteringService:
    """
    Service for clustering news articles into events.
    
    Uses combination of:
    - Semantic embeddings for similarity
    - Keyword extraction for grouping
    - Time windows for temporal clustering
    
    Integrates with Pathway streaming tables for incremental processing.
    """
    
    def __init__(self):
        self.similarity_threshold = settings.pathway.similarity_threshold
        self.event_window_hours = settings.pathway.event_window_hours
        
        # In-memory storage for clustering (Pathway-compatible)
        self._clusters: Dict[str, Dict] = {}
        self._next_cluster_id = 0
    
    def add_article(self, article: Article, embedding: List[float]) -> str:
        """
        Add an article to clustering.
        Returns the cluster ID if assigned.
        """
        article_data = {
            "article_id": article.article_id,
            "title": article.title,
            "description": article.description,
            "content": article.content,
            "source": article.source,
            "category": article.category.value,
            "published_at": article.published_at,
            "embedding": embedding,
            "keywords": self._extract_keywords(article.title + " " + (article.description or ""))
        }
        
        # Try to assign to existing cluster
        cluster_id = self._find_best_cluster(article_data)
        
        if cluster_id:
            self._add_to_cluster(cluster_id, article_data)
            return cluster_id
        else:
            # Create new cluster
            cluster_id = self._create_cluster(article_data)
            return cluster_id
    
    def _create_cluster(self, article_data: Dict[str, Any]) -> str:
        """Create a new cluster from an article"""
        self._next_cluster_id += 1
        cluster_id = f"evt_{self._next_cluster_id:06d}"
        
        self._clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "title": article_data["title"],
            "description": article_data.get("description", ""),
            "articles": [article_data],
            "embeddings": [article_data["embedding"]],
            "keywords": set(article_data["keywords"]),
            "created_at": utcnow(),
            "last_updated": utcnow(),
            "categories": {article_data["category"]},
            "sources": {article_data["source"]},
            "article_count": 1
        }
        
        logger.debug(f"Created new cluster {cluster_id}")
        return cluster_id
    
    def _add_to_cluster(self, cluster_id: str, article_data: Dict[str, Any]):
        """Add article to existing cluster"""
        cluster = self._clusters[cluster_id]
        
        cluster["articles"].append(article_data)
        cluster["embeddings"].append(article_data["embedding"])
        cluster["keywords"].update(article_data["keywords"])
        cluster["categories"].add(article_data["category"])
        cluster["sources"].add(article_data["source"])
        cluster["last_updated"] = utcnow()
        cluster["article_count"] += 1
        
        # Update cluster title if new article is more comprehensive
        if len(article_data["title"]) > len(cluster["title"]):
            cluster["title"] = article_data["title"]
        
        logger.debug(f"Added article to cluster {cluster_id}")
    
    def _find_best_cluster(self, article_data: Dict[str, Any]) -> Optional[str]:
        """Find the best matching cluster for an article"""
        best_cluster_id = None
        best_similarity = self.similarity_threshold
        
        embedding = np.array(article_data["embedding"]).reshape(1, -1)
        article_time = article_data["published_at"]
        
        for cluster_id, cluster in self._clusters.items():
            # Skip if article is too old for this cluster
            cluster_time = cluster["last_updated"]
            if hasattr(article_time, 'timestamp') and hasattr(cluster_time, 'timestamp'):
                time_diff = (cluster_time - article_time).total_seconds() / 3600
            else:
                time_diff = 0
            
            if time_diff > self.event_window_hours:
                continue
            
            # Compute similarity with cluster center
            cluster_embeddings = cluster.get("embeddings", [])
            if cluster_embeddings:
                center = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
                similarity = cosine_similarity(embedding, center)[0][0]
                
                # Check keyword overlap
                keyword_overlap = self._compute_keyword_overlap(
                    article_data["keywords"],
                    cluster["keywords"]
                )
                
                # Combined score
                combined_score = (similarity * 0.7) + (keyword_overlap * 0.3)
                
                if combined_score > best_similarity:
                    best_similarity = combined_score
                    best_cluster_id = cluster_id
        
        return best_cluster_id
    
    def _compute_keyword_overlap(
        self, 
        article_keywords: set, 
        cluster_keywords: set
    ) -> float:
        """Compute keyword overlap between article and cluster"""
        if not article_keywords or not cluster_keywords:
            return 0.0
        
        overlap = article_keywords & cluster_keywords
        union = article_keywords | cluster_keywords
        
        return len(overlap) / len(union) if union else 0.0
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text"""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Common stop words to filter
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "been",
            "with", "they", "this", "that", "from", "will", "would", "there",
            "their", "what", "when", "where", "who", "which", "after", "more",
            "about", "into", "over", "such", "than", "them", "some", "could",
            "make", "just", "only", "other", "than", "then", "these", "those"
        }
        
        word_counts = defaultdict(int)
        for word in words:
            if word not in stop_words:
                word_counts[word] += 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        return set(word for word, count in sorted_words[:10])
    
    def get_cluster_info(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Get full information about a cluster"""
        return self._clusters.get(cluster_id)
    
    def get_all_clusters(self) -> Dict[str, Dict]:
        """Get all current clusters"""
        return self._clusters
    
    def get_cluster_articles(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Get all articles in a cluster"""
        cluster = self._clusters.get(cluster_id)
        if cluster:
            return cluster["articles"]
        return []
    
    def compute_cluster_similarity_matrix(self, cluster_id: str) -> np.ndarray:
        """Compute pairwise similarity matrix for articles in a cluster"""
        cluster = self._clusters.get(cluster_id)
        if not cluster or len(cluster["articles"]) < 2:
            return np.array([])
        
        embeddings = np.array(cluster["embeddings"])
        return cosine_similarity(embeddings)
    
    def detect_major_update(
        self, 
        cluster_id: str, 
        new_article: Dict[str, Any]
    ) -> bool:
        """
        Detect if a new article represents a major update.
        """
        cluster = self._clusters.get(cluster_id)
        if not cluster:
            return True
        
        # Check similarity with recent articles
        recent_articles = cluster["articles"][-5:]
        similarities = []
        
        new_embedding = np.array(new_article["embedding"]).reshape(1, -1)
        
        for article in recent_articles:
            old_embedding = np.array(article["embedding"]).reshape(1, -1)
            sim = cosine_similarity(new_embedding, old_embedding)[0][0]
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        
        # Low similarity indicates significant new angle or information
        if avg_similarity < 0.5:
            return True
        
        # Check for correction keywords
        correction_keywords = [
            "correction", "update", "amended", "revised", "clarification",
            "retraction", "debunked", "false", "misinformation"
        ]
        
        text = (new_article.get("title", "") + " " + 
                new_article.get("description", "")).lower()
        
        for keyword in correction_keywords:
            if keyword in text:
                return True
        
        return False
    
    def cleanup_inactive_clusters(self, max_hours: int = None) -> int:
        """Remove clusters that haven't been updated recently"""
        if max_hours is None:
            max_hours = self.event_window_hours
        
        cutoff = utcnow()
        
        to_remove = []
        for cluster_id, cluster in self._clusters.items():
            last_updated = cluster.get("last_updated", utcnow())
            if hasattr(last_updated, 'timestamp'):
                hours_diff = (cutoff - last_updated).total_seconds() / 3600
            else:
                hours_diff = 0
            
            if hours_diff > max_hours:
                to_remove.append(cluster_id)
        
        for cluster_id in to_remove:
            del self._clusters[cluster_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive clusters")
        
        return len(to_remove)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about current clustering state"""
        total_articles = sum(c["article_count"] for c in self._clusters.values())
        
        return {
            "total_clusters": len(self._clusters),
            "total_clustered_articles": total_articles,
            "avg_cluster_size": total_articles / len(self._clusters) if self._clusters else 0
        }
    
    def sync_with_pathway_tables(self):
        """Sync clustering state with Pathway tables"""
        # Get events from Pathway tables
        pathway_events = pathway_tables.get_all_events()
        
        for event_id, event_data in pathway_events.items():
            if event_id not in self._clusters:
                # Import event from Pathway
                self._clusters[event_id] = {
                    "cluster_id": event_id,
                    "title": event_data.get("title", ""),
                    "description": event_data.get("description", ""),
                    "articles": [],
                    "embeddings": event_data.get("embeddings", []),
                    "keywords": set(),
                    "created_at": event_data.get("first_article_at", utcnow()),
                    "last_updated": event_data.get("last_updated_at", utcnow()),
                    "categories": set(event_data.get("categories", [])),
                    "sources": set(event_data.get("sources", [])),
                    "article_count": event_data.get("article_count", 0)
                }
                logger.info(f"Imported event {event_id} from Pathway tables")


# Global clustering service instance
clustering_service = EventClusteringService()

