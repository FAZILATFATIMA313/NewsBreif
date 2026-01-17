# LiveBrief - Real-Time News Intelligence System

**Live news clustering + AI impact analysis with RAG**

A real-time news intelligence system that continuously ingests live news, clusters related articles into events, and generates AI-powered impact analysis using RAG (Retrieval-Augmented Generation).

---

## Quick Start

```bash
cd /root/NewsBreif

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your NewsData.io and Gemini API keys

# Run full system
python main.py full

# Open browser: http://localhost:8501 (UI) and http://localhost:8000/docs (API)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LIVEBRIEF SYSTEM                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1️⃣  CONNECTORS                    2️⃣  STREAMING (Pathway)          │
│  ┌──────────────────────┐         ┌──────────────────────────────┐ │
│  │  NewsData.io API     │────────▶│  Streaming Tables            │ │
│  │  - Live news fetch   │         │  - Articles Table            │ │
│  │  - Multi-category    │         │  - Events Table              │ │
│  │  - Real-time polling │         │  - Impact Table              │ │
│  └──────────────────────┘         │  - Timeline Table            │ │
│                                    └──────────────────────────────┘ │
│                                               │                      │
│                                               ▼                      │
│  3️⃣  TRANSFORMS                    4️⃣  CLUSTERING                  │
│  ┌──────────────────────┐         ┌──────────────────────────────┐ │
│  │  - Text embeddings   │         │  Semantic Clustering         │ │
│  │  - Keyword extraction│         │  - Cosine similarity         │ │
│  │  - Feature eng.      │         │  - Keyword overlap           │ │
│  └──────────────────────┘         │  - Event assignment          │ │
│                                    └──────────────────────────────┘ │
│                                               │                      │
│                                               ▼                      │
│  5️⃣  RAG + AI                     6️⃣  FRONTEND                     │
│  ┌──────────────────────┐         ┌──────────────────────────────┐ │
│  │  - Vector store      │         │  Streamlit UI                │ │
│  │  - Impact generation │         │  - Live news cards           │ │
│  │  - Question answering│         │  - Expandable details        │ │
│  │  - Gemini LLM        │         │  - AI Q&A per event          │ │
│  └──────────────────────┘         │  - Auto-refresh              │ │
│                                    └──────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
/root/NewsBreif/
├── main.py                    # Entry point (api/ui/full modes)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .env.example               # Environment template
│
├── config/
│   └── settings.py            # Configuration management
│
├── connectors/
│   └── newsdata_connector.py  # NewsData.io API connector ⭐
│       - Fetches live news
│       - Parses article data
│       - Multi-category support
│
├── streaming/
│   ├── tables.py              # Pathway streaming tables ⭐
│   │   - PathwayStreamingTables class
│   │   - Event clustering state
│   │   - Statistics tracking
│   │
│   └── transforms.py          # Streaming transformations ⭐
│       - Embedding generation
│       - Text processing
│
├── services/
│   ├── event_clustering.py    # Clustering service ⭐
│   │   - Article clustering logic
│   │   - Similarity computation
│   │
│   ├── impact_generator.py    # AI impact analysis ⭐
│   │   - RAG-based impact generation
│   │   - Gemini LLM integration
│   │
│   ├── rag_service.py         # RAG pipeline ⭐
│   │   - Vector store management
│   │   - Context retrieval
│   │   - Question answering
│   │
│   └── api_optimizer.py       # Caching & throttling
│
├── models/
│   └── schemas.py             # Pydantic data models
│
├── api/
│   └── main.py                # FastAPI backend ⭐
│       - REST endpoints
│       - RAG endpoints
│       - Background refresh
│
└── ui/
    └── app.py                 # Streamlit frontend ⭐
        - News cards UI
        - Expandable details
        - Interactive Q&A
```

---

## Key Components

### 1. Pathway Connectors (`connectors/newsdata_connector.py`)
Fetches live news from NewsData.io API:
```python
class NewsDataConnector:
    def fetch_news(self) -> List[Article]: ...
    def fetch_new_articles(self) -> List[Article]: ...
    def parse_article(self, article_data) -> Article: ...
```

### 2. Pathway Tables (`streaming/tables.py`)
Stores and manages streaming data:
```python
class PathwayStreamingTables:
    def add_article(self, article_data): ...      # Add article
    def get_all_events(self) -> Dict: ...          # Get clustered events
    def generate_impact_for_event(self, id): ...   # Generate AI impact
    def get_cluster_stats(self) -> Dict: ...       # Statistics
```

### 3. Clustering (`services/event_clustering.py`)
Clusters related articles using semantic similarity:
```python
class ClusteringService:
    def add_article(self, article, embedding): ...  # Assign to cluster
    def get_all_events(self) -> Dict: ...            # Get events
    def cleanup_inactive_clusters(self): ...         # Remove old events
```

### 4. RAG System (`services/rag_service.py`)
Retrieval-Augmented Generation pipeline:
```python
class RAGService:
    def index_article(self, article, event_id): ...      # Store embedding
    def get_event_context(self, event_id) -> List: ...   # Retrieve
    def answer_question(self, event, question) -> str: ... # Generate
```

### 5. Impact Generator (`services/impact_generator.py`)
AI-powered impact analysis:
```python
class ImpactGenerator:
    def generate_impact(self, event) -> ImpactAnalysis: ...
    # Generates: affected_groups, daily_life_impact, short_term_risk, what_to_know
```

---

## API Endpoints

### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/events` | List all news events |
| GET | `/api/v1/events/{id}` | Get event details |
| GET | `/api/v1/events/{id}/impact` | Get AI impact analysis |
| GET | `/api/v1/events/{id}/timeline` | Get event timeline |
| GET | `/api/v1/search?q=query` | Search events |
| GET | `/api/v1/stats` | Get system statistics |

### RAG Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/events/{id}/context` | Get retrieved articles |
| POST | `/api/v1/events/{id}/query` | Ask AI about event |
| POST | `/api/v1/events/{id}/regenerate-impact` | Regenerate analysis |

---

## Configuration

Create `.env` file:
```env
# NewsData.io API (https://newsdata.io/)
NEWSDATA_API_KEY=your_key_here
NEWSDATA_LANGUAGE=en
NEWSDATA_COUNTRY=us
NEWSDATA_CATEGORIES=technology,business,health,science,entertainment,sports

# Gemini API (https://aistudio.google.com/)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3-flash

# Pathway Settings
PATHWAY_POLLING_INTERVAL=60
PATHWAY_SIMILARITY_THRESHOLD=0.75
```

---

## Running Modes

```bash
# Full system (API + UI + background refresh)
python main.py full

# API server only
python main.py api

# UI only (connects to existing API)
python main.py ui
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Streaming | Pathway | Incremental data processing |
| News Source | NewsData.io API | Real-time news data |
| AI | Google Gemini + RAG | Impact analysis, Q&A |
| Backend | FastAPI | REST API |
| Frontend | Streamlit | Real-time UI |
| Data | In-memory | Vector store, tables |

---


Built with Pathway, NewsData.io, and Google Gemini

