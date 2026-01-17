"""
Impact Generator Service for LiveBrief.
Automatically generates impact analysis for news events.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import json
import re
from loguru import logger

import google.genai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from models.schemas import ImpactAnalysis, NewsEvent

# Import API optimizer for caching
try:
    from services.api_optimizer import api_cache
    GEMINI_CACHE_ENABLED = True
except ImportError:
    GEMINI_CACHE_ENABLED = False
    logger.warning("API optimizer not available, caching disabled")


def utcnow() -> datetime:
    """Get current UTC time with timezone info (timezone-aware)"""
    return datetime.now(timezone.utc)


def _extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract and parse JSON from potentially malformed text"""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # Find JSON boundaries
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1 or end <= start:
        return None
    
    # Extract and try to fix
    json_str = text[start:end+1]
    
    # Try 1: Direct parse of extracted
    try:
        return json.loads(json_str)
    except:
        pass
    
    # Try 2: Fix unterminated quotes line by line
    lines = json_str.split('\n')
    for i, line in enumerate(lines):
        # Count quotes that aren't escaped
        quote_count = 0
        j = 0
        while j < len(line):
            if line[j] == '\\' and j + 1 < len(line):
                j += 2
                continue
            if line[j] == '"':
                quote_count += 1
            j += 1
        if quote_count % 2 == 1:
            lines[i] = line + '"'
    
    fixed = '\n'.join(lines)
    try:
        return json.loads(fixed)
    except:
        pass
    
    # Try 3: More aggressive quote fixing
    fixed2 = fixed.replace('\n', ' ').replace('  ', ' ')
    try:
        return json.loads(fixed2)
    except:
        pass
    
    return None


class ImpactGenerator:
    """
    Generates automatic impact analysis for news events.
    
    Impact dimensions:
    - Who is affected
    - Short-term daily life impact
    - Potential risks
    - Key information people should know
    """
    
    def __init__(self):
        self.api_key = settings.gemini.api_key
        self.model = settings.gemini.model
        self.max_tokens = settings.gemini.max_output_tokens
        self.temperature = settings.gemini.temperature
        self.timeout = settings.gemini.timeout
        
        # Configure Gemini client
        self._client = genai.Client(api_key=self.api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_impact(self, event: NewsEvent) -> ImpactAnalysis:
        """
        Generate impact analysis for a news event.
        Uses caching to avoid redundant Gemini API calls.
        
        Args:
            event: The news event to analyze
            
        Returns:
            ImpactAnalysis with computed impact dimensions
        """
        # Check cache first to avoid API calls
        cache_key = f"impact:{event.event_id}"
        if GEMINI_CACHE_ENABLED:
            cached = api_cache.get(cache_key)
            if cached:
                logger.debug(f"Using cached impact for {event.event_id}")
                return ImpactAnalysis(**cached)
        
        # Build context from articles
        context = self._build_event_context(event)
        
        # Generate impact using LLM
        impact_data = self._call_llm(context)
        
        if impact_data:
            impact_analysis = ImpactAnalysis(
                affected_groups=impact_data.get("affected_groups", []),
                daily_life_impact=impact_data.get("daily_life_impact", "No significant impact"),
                short_term_risk=impact_data.get("short_term_risk", "Low risk"),
                what_to_know=impact_data.get("what_to_know", ""),
                generated_at=utcnow(),
                confidence=impact_data.get("confidence", 0.8)
            )
            
            # Cache the result
            if GEMINI_CACHE_ENABLED:
                api_cache.set(cache_key, {
                    "affected_groups": impact_analysis.affected_groups,
                    "daily_life_impact": impact_analysis.daily_life_impact,
                    "short_term_risk": impact_analysis.short_term_risk,
                    "what_to_know": impact_analysis.what_to_know,
                    "generated_at": str(impact_analysis.generated_at),
                    "confidence": impact_analysis.confidence
                })
            
            return impact_analysis
        else:
            # Fallback to simple analysis
            return self._generate_simple_impact(event)
    
    def _build_event_context(self, event: NewsEvent) -> str:
        """Build context string from event for LLM"""
        articles_info = []
        
        for article_id in event.article_ids[:10]:  # Limit to 10 articles
            articles_info.append(f"- Article ID: {article_id}")
        
        categories = ", ".join(event.categories)
        sources = ", ".join(event.sources)
        
        context = f"""
# News Event Analysis Request

## Event Information
- Event ID: {event.event_id}
- Title: {event.title}
- Description: {event.description}
- Categories: {categories}
- Sources: {sources}
- Total Articles: {len(event.article_ids)}
- Sentiment Trend: {event.sentiment_trend}
- Article Count: {event.article_count}

## Live Summary
{event.live_summary or "No summary available."}

## Articles
{chr(10).join(articles_info)}

## Task
Generate impact analysis for this news event. Consider:
1. Who/what groups are affected by this event?
2. How might this affect people's daily life in the short term?
3. What potential risks or concerns should people be aware of?
4. What key information should people know about this event?

Please provide your analysis in the following JSON format:
{{
    "affected_groups": ["list", "of", "affected", "groups"],
    "daily_life_impact": "brief description of short-term daily life impact",
    "short_term_risk": "brief description of potential short-term risks",
    "what_to_know": "brief advice or key information",
    "confidence": 0.0-1.0
}}
"""
        return context
    
    def _call_llm(self, context: str) -> Optional[Dict[str, Any]]:
        """Call LLM to generate impact analysis with robust JSON parsing"""
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=context,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json"
                }
            )
            
            # Get response text
            response_text = response.text
            
            # Clean up markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Try to parse with robust extraction
            impact_data = _extract_json_from_text(response_text)
            
            if impact_data:
                logger.info(f"Generated impact analysis for event using {self.model}")
                return impact_data
            else:
                logger.warning(f"Failed to parse LLM response, will use simple analysis")
                return None
            
        except Exception as e:
            logger.warning(f"Gemini API error ({self.model}): {type(e).__name__}. Using simple analysis.")
            return None
    
    def _generate_simple_impact(self, event: NewsEvent) -> ImpactAnalysis:
        """
        Generate simple impact analysis without LLM.
        Used as fallback when LLM is unavailable.
        """
        # Simple keyword-based impact detection
        title_lower = event.title.lower()
        description_lower = (event.description or "").lower()
        
        affected_groups = []
        daily_life_impact = []
        risks = []
        
        # Check for common impact patterns
        impact_keywords = {
            "transportation": ["commuters", "travelers", "drivers", "public transit"],
            "economy": ["consumers", "businesses", "investors", "shoppers"],
            "health": ["patients", "healthcare", "residents", "families"],
            "technology": ["users", "businesses", "developers", "consumers"],
            "environment": ["residents", "communities", "wildlife", "travelers"],
            "politics": ["citizens", "voters", "businesses", "residents"]
        }
        
        text = title_lower + " " + description_lower
        
        for category, keywords in impact_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if keyword not in affected_groups:
                        affected_groups.append(keyword)
        
        if not affected_groups:
            affected_groups = ["general public"]
        
        # Simple risk assessment based on sentiment
        if event.sentiment_trend == "negative":
            short_term_risk = "Monitor developments, potential disruptions expected"
        elif event.sentiment_trend == "positive":
            short_term_risk = "Low risk, generally positive developments"
        else:
            short_term_risk = "Uncertain, stay informed"
        
        daily_life_impact = f"May affect {', '.join(affected_groups[:2])} in the near term"
        
        return ImpactAnalysis(
            affected_groups=affected_groups,
            daily_life_impact=daily_life_impact,
            short_term_risk=short_term_risk,
            what_to_know=f"Event has {event.article_count} related articles from {len(event.sources)} sources. Monitor for updates.",
            generated_at=utcnow(),
            confidence=0.5
        )
    
    def update_impact(
        self, 
        existing_impact: ImpactAnalysis, 
        event: NewsEvent
    ) -> ImpactAnalysis:
        """
        Update existing impact analysis with new information.
        Called when event evolves with new articles.
        """
        # Check if update is needed
        time_since_last = (utcnow() - existing_impact.generated_at).total_seconds()
        
        # Update if more than 1 hour since last analysis or significant change
        if time_since_last > 3600 or event.article_count > existing_impact.confidence * 10:
            return self.generate_impact(event)
        
        # Otherwise, return existing with updated timestamp
        return ImpactAnalysis(
            affected_groups=existing_impact.affected_groups,
            daily_life_impact=existing_impact.daily_life_impact,
            short_term_risk=existing_impact.short_term_risk,
            what_to_know=existing_impact.what_to_know,
            generated_at=utcnow(),
            confidence=existing_impact.confidence
        )
    
    def generate_batch_impacts(
        self, 
        events: List[NewsEvent]
    ) -> Dict[str, ImpactAnalysis]:
        """
        Generate impact analysis for multiple events.
        Useful for initial batch processing.
        """
        results = {}
        
        for event in events:
            try:
                impact = self.generate_impact(event)
                results[event.event_id] = impact
                logger.info(f"Generated impact for event {event.event_id}")
            except Exception as e:
                logger.error(f"Failed to generate impact for event {event.event_id}: {e}")
        
        return results


# Global impact generator instance
impact_generator = ImpactGenerator()

