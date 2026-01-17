"""
FastAPI Backend for LiveBrief.
Provides API endpoints for news events, impact analysis, and system status.
"""
import sys
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from config.settings import settings
from streaming.tables import pathway_tables
from connectors.newsdata_connector import create_newsdata_connector
from streaming.transforms import transforms
from services.event_clustering import clustering_service
from services.rag_service import rag_service
from services.impact_generator import impact_generator
from models.schemas import APIResponse


def utcnow() -> datetime:
    """Get current UTC time with timezone info"""
    return datetime.now(timezone.utc)


# Global state
_connector = None
_refresh_thread: threading.Thread = None
_running = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global _connector, _refresh_thread, _running
    
    logger.info("Starting LiveBrief API server")
    
    # Initialize Pathway tables
    pathway_tables.initialize()
    
    # Initialize connector
    _connector = create_newsdata_connector()
    
    # Fetch initial articles
    logger.info("Fetching initial news articles...")
    try:
        initial_articles = _connector.fetch_new_articles()
        logger.info(f"Fetched {len(initial_articles)} initial articles")
        
        for article in initial_articles:
            article_dict = article.dict()
            processed = transforms.process_article(article_dict)
            pathway_tables.add_article(processed)
            
            event_id = processed.get("event_id")
            if event_id:
                try:
                    rag_service.index_article(article, event_id)
                except Exception as e:
                    logger.warning(f"Could not index article in RAG: {e}")
        
        logger.info(f"Indexed {len(initial_articles)} articles for RAG")
        
    except Exception as e:
        logger.error(f"Error fetching initial articles: {e}")
    
    # Start background news refresh thread
    def news_refresh_worker():
        """Background worker for news refresh"""
        logger.info("Starting news refresh worker in API mode")
        interval = settings.pathway.polling_interval
        
        while _running:
            try:
                articles = _connector.fetch_new_articles()
                
                for article in articles:
                    article_dict = article.dict()
                    processed = transforms.process_article(article_dict)
                    pathway_tables.add_article(processed)
                    
                    embedding = processed.get("embedding", [])
                    clustering_service.add_article(article, embedding)
                    
                    event_id = processed.get("event_id")
                    if event_id and settings.gemini.api_key:
                        try:
                            rag_service.index_article(article, event_id)
                        except Exception as e:
                            logger.warning(f"Could not index article in RAG: {e}")
                
                clustering_service.cleanup_inactive_clusters()
                pathway_tables.cleanup_old_events()
                
            except Exception as e:
                logger.error(f"Error in refresh worker: {e}")
            
            for _ in range(interval):
                if not _running:
                    break
                import time
                time.sleep(1)
        
        _connector._close_client()
        logger.info("News refresh worker stopped")
    
    _refresh_thread = threading.Thread(target=news_refresh_worker, daemon=True)
    _refresh_thread.start()
    
    yield
    
    logger.info("Shutting down LiveBrief API server")
    _running = False
    
    if _refresh_thread and _refresh_thread.is_alive():
        _refresh_thread.join(timeout=5)


app = FastAPI(
    title="LiveBrief API",
    description="Real-Time News Evolution & Impact Intelligence System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_time_ago(dt) -> str:
    """Format datetime as human-readable time ago"""
    try:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        now = utcnow()
        diff = (now - dt).total_seconds()
        
        if diff < 60:
            return "Just now"
        elif diff < 3600:
            return f"{int(diff/60)}m ago"
        elif diff < 86400:
            return f"{int(diff/3600)}h ago"
        else:
            return f"{int(diff/86400)}d ago"
    except:
        return "Unknown"


@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API info"""
    return APIResponse(
        success=True,
        data={
            "name": "LiveBrief API",
            "version": "1.0.0",
            "status": "running"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": utcnow().isoformat(),
        "service": "livebrief-api"
    }


@app.get("/api/v1/events")
async def get_events(limit: int = 20):
    """Get news events from pathway_tables"""
    try:
        all_events = pathway_tables.get_all_events()
        event_ids = list(all_events.keys())
        
        event_list = []
        for event_id in event_ids:
            event_data = all_events[event_id]
            
            # Generate impact if not already done
            if not event_data.get("impact_generated", False):
                try:
                    pathway_tables.generate_impact_for_event(event_id)
                    event_data = pathway_tables.get_event(event_id) or event_data
                except Exception as e:
                    logger.warning(f"Could not generate impact for {event_id}: {e}")
            
            impact = event_data.get("impact", {})
            
            event_list.append({
                "event_id": event_id,
                "title": event_data.get("title", ""),
                "description": event_data.get("description", ""),
                "headline": event_data.get("title", ""),
                "summary": event_data.get("description", ""),
                "category": event_data.get("categories", ["general"])[0] if event_data.get("categories") else "general",
                "source": event_data.get("sources", ["Unknown"])[0] if event_data.get("sources") else "Unknown",
                "time_ago": format_time_ago(event_data.get("last_updated_at", utcnow())),
                "last_updated": format_time_ago(event_data.get("last_updated_at", utcnow())),
                "article_count": event_data.get("article_count", 0),
                "article_ids": event_data.get("article_ids", []),
                "affected_groups": impact.get("affected_groups", event_data.get("categories", [])),
                "impact_summary": impact.get("daily_life_impact", event_data.get("impact_summary", "Impact analysis available")),
                "is_live": True,
                "categories": event_data.get("categories", []),
                "sources": event_data.get("sources", []),
                "last_updated_at": str(event_data.get("last_updated_at", "")),
                "impact_generated": True,
                "impact_confidence": impact.get("confidence", 0.8)
            })
        
        event_list.sort(key=lambda x: x["last_updated_at"], reverse=True)
        
        return {
            "success": True,
            "data": {
                "events": event_list[:limit],
                "total": len(event_list)
            },
            "timestamp": utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/events/{event_id}")
async def get_event(event_id: str):
    """Get detailed information about a specific event"""
    event = pathway_tables.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return {
        "success": True,
        "data": {
            "event_id": event_id,
            "title": event.get("title", ""),
            "description": event.get("description", ""),
            "article_count": event.get("article_count", 0),
            "categories": event.get("categories", []),
            "sources": event.get("sources", []),
            "article_ids": event.get("article_ids", []),
            "first_article_at": str(event.get("first_article_at", "")),
            "last_updated_at": str(event.get("last_updated_at", "")),
            "source_type": "live"
        }
    }


@app.get("/api/v1/events/{event_id}/impact")
async def get_event_impact(event_id: str):
    """Get impact analysis for an event"""
    event = pathway_tables.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    impact_data = event.get("impact", {})
    
    return {
        "success": True,
        "data": {
            "affected_groups": impact_data.get("affected_groups", event.get("categories", [])),
            "daily_life_impact": impact_data.get("daily_life_impact", event.get("impact_summary", "This event may affect daily activities in the near term.")),
            "short_term_risk": impact_data.get("short_term_risk", "Monitor developments for potential risks."),
            "what_to_know": impact_data.get("what_to_know", "Stay updated for the latest information."),
            "generated_at": str(event.get("last_updated_at", utcnow())),
            "source_type": "live"
        }
    }


@app.get("/api/v1/events/{event_id}/timeline")
async def get_event_timeline(event_id: str):
    """Get timeline of event evolution"""
    event = pathway_tables.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    article_ids = event.get("article_ids", [])
    timeline = []
    
    for i, article_id in enumerate(article_ids):
        timeline.append({
            "update_id": f"{event_id}_{article_id}",
            "event_id": event_id,
            "article_id": article_id,
            "update_type": "new" if i == 0 else "update",
            "summary": f"Article {article_id}",
            "published_at": str(event.get("first_article_at", "")),
            "is_major": i == 0
        })
    
    return {
        "success": True,
        "data": {
            "event_id": event_id,
            "timeline": timeline,
            "source_type": "live"
        }
    }


@app.get("/api/v1/search")
async def search_events(q: str = Query(...)):
    """Search events by keywords"""
    all_events = pathway_tables.get_all_events()
    event_ids = list(all_events.keys())
    query = q.lower()
    
    results = []
    for event_id in event_ids:
        event_data = all_events[event_id]
        title = event_data.get("title", "").lower()
        description = event_data.get("description", "").lower()
        
        if query in title or query in description:
            results.append({
                "event_id": event_id,
                "title": event_data.get("title", ""),
                "description": event_data.get("description", "")
            })
    
    return {
        "success": True,
        "data": {
            "query": q,
            "results": results,
            "total": len(results)
        }
    }


@app.post("/api/v1/admin/refresh")
async def refresh_news():
    """Trigger a manual news refresh"""
    try:
        connector = create_newsdata_connector()
        articles = connector.fetch_new_articles()
        
        for article in articles:
            article_dict = article.dict()
            processed = transforms.process_article(article_dict)
            pathway_tables.add_article(processed)
        
        return {
            "success": True,
            "data": {
                "articles_processed": len(articles)
            },
            "message": f"Processed {len(articles)} articles"
        }
        
    except Exception as e:
        logger.error(f"Error refreshing news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "success": True,
        "data": pathway_tables.get_cluster_stats()
    }


@app.get("/api/v1/events/{event_id}/context")
async def get_event_context(event_id: str):
    """Get context articles used to generate impact for an event"""
    event = pathway_tables.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    context_articles = rag_service.get_event_context(event_id, max_articles=10)
    
    articles = []
    for article in context_articles:
        articles.append({
            "article_id": article.article_id,
            "title": article.title,
            "source": article.source,
            "published_at": article.published_at.isoformat() if hasattr(article.published_at, 'isoformat') else str(article.published_at),
            "description": article.description or ""
        })
    
    return {
        "success": True,
        "data": {
            "event_id": event_id,
            "event_title": event.get("title", ""),
            "retrieved_articles": articles,
            "total_retrieved": len(articles),
            "retrieval_method": "RAG - cosine similarity over article embeddings",
            "note": "These articles were retrieved and used to generate the impact analysis",
            "source_type": "live"
        },
        "timestamp": utcnow().isoformat()
    }


@app.post("/api/v1/events/{event_id}/query")
async def query_event_with_rag(event_id: str, request: Dict[str, str]):
    """Query an event using RAG to get a personalized answer"""
    event = pathway_tables.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    question = request.get("question", "Why does this event matter?")
    
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
        article_count=event.get("article_count", 0)
    )
    
    answer = rag_service.answer_question(news_event, question)
    context_articles = rag_service.get_event_context(event_id, max_articles=5)
    sources = [{"title": a.title, "source": a.source} for a in context_articles]
    
    return {
        "success": True,
        "data": {
            "event_id": event_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "method": "RAG - Retrieved relevant articles, then generated answer using Gemini",
            "dynamism_note": "This answer is generated on-demand using the latest indexed articles",
            "source_type": "live"
        },
        "timestamp": utcnow().isoformat()
    }


@app.post("/api/v1/events/{event_id}/regenerate-impact")
async def regenerate_impact(event_id: str):
    """Regenerate impact analysis for an event using RAG"""
    event = pathway_tables.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
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
        article_count=event.get("article_count", 0)
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
    
    pathway_tables._clustering_state[event_id]["impact"] = impact_data
    pathway_tables._clustering_state[event_id]["impact_generated"] = True
    
    return {
        "success": True,
        "data": {
            "event_id": event_id,
            "impact": impact_data,
            "dynamism": {
                "regenerated": True,
                "previous_article_count": event.get("article_count", 0),
                "retrieval_method": "RAG over indexed articles",
                "generation_model": settings.gemini.model,
                "note": "Impact was regenerated using the latest RAG context"
            },
            "source_type": "live"
        },
        "timestamp": utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.app.port)

