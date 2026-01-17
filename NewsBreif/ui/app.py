"""
LiveBrief Streamlit Frontend.
Real-time news intelligence UI with expandable news cards.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import time

st.set_page_config(
    page_title="LiveBrief - Real-Time News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"


def utcnow() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)


def init_session_state():
    """Initialize session state"""
    if 'refresh_enabled' not in st.session_state:
        st.session_state.refresh_enabled = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 30


def fetch_events(limit: int = 50) -> Dict[str, Any]:
    """Fetch events from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/events", params={"limit": limit}, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching events: {e}")
        return {"data": {"events": []}}


def fetch_stats() -> Dict[str, Any]:
    """Fetch system stats from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return {"data": {"total_events": 0, "total_clustered_articles": 0}}


def fetch_event_context(event_id: str) -> Dict[str, Any]:
    """Fetch RAG context for an event"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/events/{event_id}/context", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return {"data": {}}


def query_event_with_rag(event_id: str, question: str) -> Dict[str, Any]:
    """Query an event using RAG"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/events/{event_id}/query",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def format_time_ago(dt_str: str) -> str:
    """Format datetime string"""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
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


def get_category_emoji(category: str) -> str:
    """Get emoji for category"""
    category_emojis = {
        "business": "💼",
        "technology": "💻",
        "entertainment": "🎬",
        "sports": "⚽",
        "politics": "🏛️",
        "health": "🏥",
        "science": "🔬",
        "world": "🌍",
        "general": "📰"
    }
    return category_emojis.get(category.lower(), "📰")


def get_impact_color(article_count: int) -> str:
    """Get color based on article count"""
    if article_count >= 5:
        return "🔴"
    elif article_count >= 3:
        return "🟠"
    else:
        return "🟢"


def render_news_card(event: Dict[str, Any]) -> None:
    """Render an expandable news card with improved layout"""
    event_id = event.get("event_id", "")
    headline = event.get("headline", "No headline")
    summary = event.get("summary", "")
    category = event.get("category", "general").upper()
    source = event.get("source", "Unknown")
    time_ago = event.get("time_ago", "")
    article_count = event.get("article_count", 1)
    impact_summary = event.get("impact_summary", "")
    affected_groups = event.get("affected_groups", [])
    
    with st.container(border=True):
        # Header row
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            emoji = get_category_emoji(category)
            st.markdown(f"**{emoji} {category}**")
        with col2:
            importance = get_impact_color(article_count)
            st.markdown(f"{importance} **{article_count}** article{'s' if article_count > 1 else ''}")
        with col3:
            if time_ago:
                st.markdown(f"<div style='text-align: right; color: #666;'>🕐 {time_ago}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Headline
        st.markdown(f"### {headline}")
        st.caption(f"📡 {source}")
        
        # Expandable details
        with st.expander("📖 Read more", expanded=False):
            if summary:
                st.markdown("**Summary**")
                st.write(summary)
            
            st.divider()
            
            if impact_summary:
                st.markdown("#### 📍 Impact on Daily Life")
                st.info(impact_summary)
            
            if affected_groups:
                st.markdown("**Affected Groups:** " + ", ".join([f"`{g}`" for g in affected_groups[:5]]))
            
            st.divider()
            
            # RAG Query section
            st.markdown("#### 🤖 Ask about this event")
            
            col_q1, col_q2 = st.columns([4, 1])
            with col_q1:
                question = st.text_input(
                    "Ask:",
                    value="Why does this event matter?",
                    key=f"q_{event_id}",
                    label_visibility="collapsed"
                )
            with col_q2:
                if st.button("Ask", key=f"ask_{event_id}", use_container_width=True):
                    with st.spinner("Thinking..."):
                        result = query_event_with_rag(event_id, question)
                        if "error" not in result:
                            answer = result.get("data", {}).get("answer", "")
                            if answer:
                                st.markdown("**Answer:**")
                                st.write(answer)
                        else:
                            st.error(f"Error: {result.get('error')}")
            
            # Source articles
            if st.button("📚 View Source Articles", key=f"src_{event_id}"):
                with st.spinner("Loading..."):
                    context = fetch_event_context(event_id)
                    data = context.get("data", {})
                    articles = data.get("retrieved_articles", [])
                    
                    if articles:
                        st.markdown(f"**{len(articles)} source articles:**")
                        for i, article in enumerate(articles):
                            with st.expander(f"📰 {article.get('title', 'Unknown')[:70]}...", expanded=i==0):
                                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                                st.write(f"**Description:** {article.get('description', 'No description')}")
                    else:
                        st.info("No source articles indexed yet.")


def render_stats_row(stats: Dict[str, Any]) -> None:
    """Render stats in a row"""
    data = stats.get("data", {})
    total_events = data.get("total_events", 0)
    total_articles = data.get("total_clustered_articles", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📰 Events", total_events)
    with col2:
        st.metric("📄 Articles", total_articles)
    with col3:
        avg = round(total_articles / max(total_events, 1), 1)
        st.metric("📊 Avg/Event", avg)
    with col4:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("🟢 API Connected")
            else:
                st.error("🔴 API Error")
        except:
            st.error("🔴 API Unreachable")


def main():
    """Main app"""
    init_session_state()
    
    st.title("📰 LiveBrief")
    st.caption("Real-Time News Intelligence with Impact Analysis")
    
    # Sidebar
    with st.sidebar:
        st.title("🔄 Auto-Refresh")
        
        st.session_state.refresh_enabled = st.toggle(
            "Enable auto-refresh",
            value=st.session_state.refresh_enabled
        )
        
        if st.session_state.refresh_enabled:
            st.session_state.refresh_interval = st.select_slider(
                "Interval (seconds)",
                options=[15, 30, 60, 120],
                value=st.session_state.refresh_interval
            )
        
        st.divider()
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        st.info("""
        **How it works:**
        - News articles are fetched continuously
        - Related articles are clustered into events
        - AI generates impact analysis for each event
        - Ask questions about any event
        """)
    
    # Stats row
    render_stats_row(fetch_stats())
    
    st.divider()
    
    # Events section
    st.markdown("### 📰 Latest News Events")
    
    events_data = fetch_events(limit=50)
    events = events_data.get("data", {}).get("events", [])
    
    if events:
        st.success(f"Found {len(events)} events from news clustering")
        
        for event in events:
            render_news_card(event)
    else:
        st.info("Waiting for news... Make sure the API server is running with `python main.py api`")
    
    # Auto-refresh
    if st.session_state.refresh_enabled:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()

