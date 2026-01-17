"""
RAG Service for LiveBrief.
Provides Retrieval-Augmented Generation for live summaries and context.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import json
from loguru import logger

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from models.schemas import NewsEvent, Article


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service for LiveBrief.
    
    Provides:
    - Live summary generation for events
    - Context retrieval from article history
    - Answer generation for event questions
    """
    
    def __init__(self):
        self.api_key = settings.gemini.api_key
        self.model = settings.gemini.model
        self.max_tokens = settings.gemini.max_output_tokens
        self.temperature = settings.gemini.temperature
        self.timeout = settings.gemini.timeout
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model)
        
        # In-memory vector store (can be replaced with actual vector DB)
        self._article_store: Dict[str, Article] = {}
        self._event_articles: Dict[str, List[str]] = {}  # event_id -> article_ids
    
    def index_article(self, article: Article, event_id: str):
        """
        Index an article for RAG retrieval.
        
        Args:
            article: Article to index
            event_id: Event this article belongs to
        """
        self._article_store[article.article_id] = article
        
        if event_id not in self._event_articles:
            self._event_articles[event_id] = []
        
        # Add to event's article list (keep in order)
        if article.article_id not in self._event_articles[event_id]:
            self._event_articles[event_id].append(article.article_id)
    
    def get_event_context(
        self, 
        event_id: str, 
        max_articles: int = 10
    ) -> List[Article]:
        """
        Get context articles for an event.
        
        Args:
            event_id: Event to get context for
            max_articles: Maximum number of articles to return
            
        Returns:
            List of Article objects
        """
        article_ids = self._event_articles.get(event_id, [])
        
        # Get most recent articles
        recent_ids = article_ids[-max_articles:]
        
        articles = []
        for aid in recent_ids:
            if aid in self._article_store:
                articles.append(self._article_store[aid])
        
        return articles
    
    def generate_live_summary(self, event: NewsEvent) -> str:
        """
        Generate a live summary for a news event.
        
        Args:
            event: The news event to summarize
            
        Returns:
            Live summary string
        """
        # Get context articles
        context_articles = self.get_event_context(event.event_id)
        
        if not context_articles:
            return event.description or ""
        
        # Build context for LLM
        context = self._build_summary_context(event, context_articles)
        
        # Generate summary
        summary = self._call_llm_summary(context)
        
        return summary if summary else self._generate_simple_summary(event)
    
    def _build_summary_context(
        self, 
        event: NewsEvent, 
        articles: List[Article]
    ) -> str:
        """Build context string for summary generation"""
        article_summaries = []
        
        for i, article in enumerate(articles):
            summary = f"""
## Article {i+1}
- Source: {article.source}
- Published: {article.published_at.strftime('%Y-%m-%d %H:%M')}
- Title: {article.title}
- Description: {article.description or 'No description'}
"""
            article_summaries.append(summary)
        
        context = f"""
# Generate Live News Summary

## Event Overview
- Event: {event.title}
- Articles: {len(articles)}
- Sources: {', '.join(event.sources)}
- Categories: {', '.join(event.categories)}
- Sentiment: {event.sentiment_trend}
- First Article: {event.first_article_at}
- Last Update: {event.last_updated_at}

## Task
Generate a concise live summary (2-3 paragraphs) that captures:
1. What is happening
2. Key developments
3. Current state of the story

Do NOT include lists or bullet points. Write as flowing prose.

## Source Articles
{chr(10).join(article_summaries)}

## Output
Provide only the summary text, no JSON or markdown formatting.
"""
        return context
    
    def _call_llm_summary(self, context: str) -> Optional[str]:
        """Call LLM to generate summary"""
        try:
            response = self._model.generate_content(
                context,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
            
            summary = response.text.strip()
            logger.info("Generated live summary via RAG")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def _generate_simple_summary(self, event: NewsEvent) -> str:
        """Generate simple summary without LLM"""
        return event.description or f"Latest developments in: {event.title}"
    
    def answer_question(
        self, 
        event: NewsEvent, 
        question: str
    ) -> str:
        """
        Answer a question about an event using RAG.
        
        Args:
            event: The news event
            question: User question
            
        Returns:
            Answer string
        """
        context_articles = self.get_event_context(event.event_id)
        
        if not context_articles:
            return "I don't have enough information about this event yet."
        
        context = self._build_qa_context(event, context_articles, question)
        answer = self._call_llm_qa(context)
        
        return answer if answer else "Unable to generate answer at this time."
    
    def _build_qa_context(
        self, 
        event: NewsEvent, 
        articles: List[Article], 
        question: str
    ) -> str:
        """Build context for Q&A"""
        article_texts = []
        for article in articles[-5:]:  # Use recent 5 articles
            text = f"""
Source: {article.source} ({article.published_at})
Title: {article.title}
Content: {article.description or article.content or 'No content'}
"""
            article_texts.append(text)
        
        return f"""
# Question about News Event

## Event: {event.title}
{event.description}

## Question
{question}

## Task
Answer the question based on the article information below. 
If the answer cannot be found in the articles, say "I don't have enough information to answer this question."

## Articles
{chr(10).join(article_texts)}

## Answer
"""
    
    def _call_llm_qa(self, context: str) -> Optional[str]:
        """Call LLM to answer question"""
        try:
            response = self._model.generate_content(
                context,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for Q&A
                    "max_output_tokens": 512
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return None
    
    def get_event_timeline(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get timeline of event evolution.
        
        Args:
            event_id: Event to get timeline for
            
        Returns:
            List of timeline events with timestamp and description
        """
        article_ids = self._event_articles.get(event_id, [])
        timeline = []
        
        for aid in article_ids:
            if aid in self._article_store:
                article = self._article_store[aid]
                timeline.append({
                    "timestamp": article.published_at.isoformat(),
                    "source": article.source,
                    "title": article.title,
                    "type": "article"
                })
        
        return timeline
    
    def search_articles(
        self, 
        query: str, 
        event_id: Optional[str] = None,
        max_results: int = 5
    ) -> List[Tuple[Article, float]]:
        """
        Search articles for relevant content.
        
        Simple keyword-based search (can be enhanced with embeddings).
        
        Args:
            query: Search query
            event_id: Optional event to limit search to
            max_results: Maximum results to return
            
        Returns:
            List of (Article, relevance_score) tuples
        """
        query_words = set(query.lower().split())
        results = []
        
        # Determine which article IDs to search
        if event_id:
            article_ids = self._event_articles.get(event_id, [])
        else:
            article_ids = list(self._article_store.keys())
        
        for aid in article_ids:
            article = self._article_store[aid]
            
            # Simple keyword matching
            text = (article.title + " " + (article.description or "")).lower()
            article_words = set(text.split())
            
            overlap = query_words & article_words
            if overlap:
                score = len(overlap) / max(len(query_words), len(article_words))
                results.append((article, score))
        
        # Sort by score and limit
        results.sort(key=lambda x: -x[1])
        return results[:max_results]
    
    def cleanup_event(self, event_id: str):
        """Remove event data from RAG store"""
        article_ids = self._event_articles.get(event_id, [])
        
        for aid in article_ids:
            if aid in self._article_store:
                del self._article_store[aid]
        
        if event_id in self._event_articles:
            del self._event_articles[event_id]
        
        logger.info(f"Cleaned up RAG data for event {event_id}")


# Global RAG service instance
rag_service = RAGService()

