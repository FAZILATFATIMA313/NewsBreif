import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class NewsDataConfig(BaseModel):
    """NewsData.io API configuration"""
    api_key: str = Field(..., description="NewsData.io API key")
    base_url: str = "https://newsdata.io/api/1/news"
    language: str = "en"
    country: str = "us"
    categories: str = "technology,business,health,science,entertainment,sports,world"
    timeout: int = 30


class GeminiConfig(BaseModel):
    """Gemini API configuration"""
    api_key: str = Field(..., description="Google Gemini API key")
    model: str = "gemini-3-flash"
    max_output_tokens: int = 1024
    temperature: float = 0.7
    timeout: int = 60


class PathwayConfig(BaseModel):
    """Pathway streaming configuration"""
    polling_interval: int = 60  # seconds
    batch_size: int = 10
    similarity_threshold: float = 0.75
    event_window_hours: int = 24
    cleanup_interval: int = 300  # seconds


class AppConfig(BaseModel):
    """Application configuration"""
    name: str = "LiveBrief"
    debug: bool = False
    log_level: str = "INFO"
    port: int = 8000


class Settings(BaseSettings):
    """Main settings class that loads from .env file"""
    newsdata: NewsDataConfig
    gemini: GeminiConfig
    pathway: PathwayConfig
    app: AppConfig

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        nested_model_update_strategy = "recursive"


def load_settings() -> Settings:
    """Load settings from environment variables"""
    # Try to load from .env file if it exists
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    
    # Create config directory if needed
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Build settings from environment
    settings = Settings(
        newsdata=NewsDataConfig(
            api_key=os.getenv("NEWSDATA_API_KEY", ""),
            base_url=os.getenv("NEWSDATA_BASE_URL", "https://newsdata.io/api/1/news"),
            language=os.getenv("NEWSDATA_LANGUAGE", "en"),
            country=os.getenv("NEWSDATA_COUNTRY", "us"),
            categories=os.getenv("NEWSDATA_CATEGORIES", "technology,business,health,science,entertainment,sports,world"),
            timeout=int(os.getenv("NEWSDATA_TIMEOUT", "30"))
        ),
        gemini=GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-3-flash"),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("GEMINI_TIMEOUT", "60"))
        ),
        pathway=PathwayConfig(
            polling_interval=int(os.getenv("PATHWAY_POLLING_INTERVAL", "60")),
            batch_size=int(os.getenv("PATHWAY_BATCH_SIZE", "10")),
            similarity_threshold=float(os.getenv("PATHWAY_SIMILARITY_THRESHOLD", "0.75")),
            event_window_hours=int(os.getenv("PATHWAY_EVENT_WINDOW_HOURS", "24")),
            cleanup_interval=int(os.getenv("PATHWAY_CLEANUP_INTERVAL", "300"))
        ),
        app=AppConfig(
            name=os.getenv("APP_NAME", "LiveBrief"),
            debug=os.getenv("APP_DEBUG", "False").lower() == "true",
            log_level=os.getenv("APP_LOG_LEVEL", "INFO"),
            port=int(os.getenv("APP_PORT", "8000")),
            news_source=os.getenv("APP_NEWS_SOURCE", "newsdata")
        )
    )
    
    return settings


# Global settings instance
settings = load_settings()
