"""
Data models for LiveBrief.
Defines structures for articles, events, and impact analysis.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ArticleCategory(str, Enum):
    """News article categories"""
    GENERAL = "general"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    HEALTH = "health"
    SCIENCE = "science"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"


class Sentiment(str, Enum):
    """Article sentiment classification"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Article(BaseModel):
    """Individual news article model"""
    article_id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article headline")
    description: Optional[str] = Field(None, description="Article summary/description")
    content: Optional[str] = Field(None, description="Full article content")
    source: str = Field(..., description="News source name")
    author: Optional[str] = Field(None, description="Article author")
    url: str = Field(..., description="Original article URL")
    image_url: Optional[str] = Field(None, description="Article image URL")
    category: ArticleCategory = Field(..., description="Article category")
    language: str = Field(default="en", description="Article language")
    country: str = Field(default="us", description="Article country")
    published_at: datetime = Field(..., description="Publication timestamp")
    fetched_at: datetime = Field(default_factory=datetime.utcnow, description="When article was fetched")
    
    # Computed fields (added during processing)
    embedding: Optional[List[float]] = Field(None, description="Article embedding vector")
    sentiment: Optional[Sentiment] = Field(None, description="Detected sentiment")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ImpactAnalysis(BaseModel):
    """Automatic impact analysis for a news event"""
    affected_groups: List[str] = Field(
        default_factory=list, 
        description="Who/what is affected by this event"
    )
    daily_life_impact: str = Field(
        ..., 
        description="Short-term impact on daily life"
    )
    short_term_risk: str = Field(
        ..., 
        description="Potential short-term risks or concerns"
    )
    what_to_know: str = Field(
        ..., 
        description="Key information people should know"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="When analysis was generated"
    )
    confidence: float = Field(
        default=0.0, 
        ge=0, 
        le=1, 
        description="Confidence score of analysis"
    )


class EventUpdate(BaseModel):
    """Represents an update to an event over time"""
    update_id: str = Field(..., description="Unique update identifier")
    event_id: str = Field(..., description="Parent event ID")
    article_id: str = Field(..., description="Source article ID")
    update_type: str = Field(
        default="update", 
        description="Type of update: new, update, correction, angle"
    )
    summary: str = Field(..., description="Brief summary of this update")
    published_at: datetime = Field(..., description="Update publication time")
    is_major: bool = Field(
        default=False, 
        description="Whether this is a major update to the story"
    )


class NewsEvent(BaseModel):
    """Clustered news event that groups related articles"""
    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title (derived from articles)")
    description: str = Field(..., description="Event description/summary")
    
    # Article references
    article_ids: List[str] = Field(
        default_factory=list, 
        description="IDs of articles in this event"
    )
    primary_article_id: Optional[str] = Field(
        None, 
        description="ID of most recent/relevant article"
    )
    
    # Temporal tracking
    first_article_at: datetime = Field(
        ..., 
        description="When first article appeared"
    )
    last_updated_at: datetime = Field(
        ..., 
        description="When event was last updated"
    )
    
    # Analysis
    categories: List[str] = Field(
        default_factory=list, 
        description="Categories in this event"
    )
    sources: List[str] = Field(
        default_factory=list, 
        description="Unique news sources"
    )
    sentiment_trend: str = Field(
        default="neutral", 
        description="Overall sentiment trend"
    )
    article_count: int = Field(
        default=1, 
        description="Number of articles in event"
    )
    
    # Impact analysis (computed)
    impact: Optional[ImpactAnalysis] = Field(
        None, 
        description="Automatic impact analysis"
    )
    live_summary: Optional[str] = Field(
        None, 
        description="LLM-generated live summary"
    )
    
    # Evolution tracking
    timeline: List[EventUpdate] = Field(
        default_factory=list, 
        description="Chronological timeline of updates"
    )
    update_frequency_hours: float = Field(
        default=0.0, 
        description="Articles per hour in last 24h"
    )
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class EventCluster(BaseModel):
    """Result of clustering articles into events"""
    event_id: str
    article_ids: List[str]
    confidence: float
    cluster_center: Optional[List[float]] = None


class StreamingEvent(BaseModel):
    """Wrapper for streaming events from Pathway"""
    event_type: str = Field(..., description="Type of event: article, event_update")
    data: Dict[str, Any] = Field(..., description="Event data payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NewsCard(BaseModel):
    """Frontend news card display model"""
    event_id: str
    category: str
    source: str
    time_ago: str
    headline: str
    summary: str
    impact_summary: str
    affected_groups: List[str]
    last_updated: str
    article_count: int
    is_live: bool = True

