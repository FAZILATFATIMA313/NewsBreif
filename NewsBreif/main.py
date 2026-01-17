"""
LiveBrief - Real-Time News Evolution & Impact Intelligence System
Main entry point with CLI options for running different components.
"""
import argparse
import sys
import signal
import threading
import importlib
from datetime import datetime
from loguru import logger
import uvicorn

from config.settings import settings


# Global state
running = True


def setup_logging():
    """Configure logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.app.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    logger.add(
        "logs/livebrief_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False


def init_services():
    """Initialize all services"""
    logger.info("Initializing services...")
    
    # Initialize Pathway streaming tables
    from streaming.tables import pathway_tables
    pathway_tables.initialize()
    logger.info("Pathway streaming tables initialized")
    
    # Import services to ensure they're ready
    from services.event_clustering import clustering_service
    from services.impact_generator import impact_generator
    from services.rag_service import rag_service
    
    logger.info("All services initialized")


def news_refresh_worker(interval: int = None):
    """Background worker for news refresh"""
    if interval is None:
        interval = settings.pathway.polling_interval
    
    from connectors.newsdata_connector import create_newsdata_connector
    from streaming.tables import pathway_tables
    from streaming.transforms import transforms
    from services.event_clustering import clustering_service
    from services.rag_service import rag_service
    
    connector = create_newsdata_connector()
    
    logger.info(f"Starting news refresh worker (interval: {interval}s)")
    
    while running:
        try:
            articles = connector.fetch_new_articles()
            
            for article in articles:
                article_dict = article.dict()
                processed = transforms.process_article(article_dict)
                pathway_tables.add_article(processed)
                
                embedding = processed.get("embedding", [])
                clustering_service.add_article(article, embedding)
                
                event_id = processed.get("event_id")
                if event_id:
                    rag_service.index_article(article, event_id)
            
            clustering_service.cleanup_inactive_clusters()
            pathway_tables.cleanup_old_events()
            
        except Exception as e:
            logger.error(f"Error in refresh worker: {e}")
        
        for _ in range(interval):
            if not running:
                break
            import time
            time.sleep(1)
    
    connector._close_client()
    logger.info("News refresh worker stopped")


def run_api():
    """Run the FastAPI server"""
    logger.info(f"Starting API server on port {settings.app.port}")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.app.port,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower()
    )


def run_ui():
    """Run the Streamlit UI"""
    import subprocess
    
    logger.info("Starting Streamlit UI")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])


def run_full():
    """Run full system (API + background refresh)"""
    global refresh_thread
    refresh_thread = None
    
    logger.info("Starting LiveBrief full system")
    
    # Initialize services
    init_services()
    
    # Start background refresh thread
    refresh_thread = threading.Thread(target=news_refresh_worker, daemon=True)
    refresh_thread.start()
    
    # Run API
    run_api()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LiveBrief - Real-Time News Intelligence")
    parser.add_argument(
        "mode",
        choices=["api", "ui", "full"],
        default="full",
        help="Run mode: api (API only), ui (UI only), full (API + background refresh)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for API server"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Override settings if provided
    if args.port:
        settings.app.port = args.port
    if args.log_level:
        settings.app.log_level = args.log_level
    
    logger.info(f"Starting LiveBrief in {args.mode} mode")
    
    # Run based on mode
    if args.mode == "api":
        run_api()
    elif args.mode == "ui":
        run_ui()
    else:
        run_full()


if __name__ == "__main__":
    main()

