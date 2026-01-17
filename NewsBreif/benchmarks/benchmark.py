"""
Benchmark scripts for LiveBrief performance testing.
"""
import time
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
import random

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def benchmark_article_processing(n_articles: int = 100):
    """Benchmark article ingestion throughput"""
    from streaming.transforms import transforms
    
    print(f"=== Article Processing Benchmark ({n_articles} articles) ===\n")
    
    # Generate test articles
    categories = ["technology", "business", "health", "science", "entertainment"]
    sources = ["Tech Daily", "Business Weekly", "Health News", "Science Today", "Entertainment Weekly"]
    
    test_articles = []
    for i in range(n_articles):
        test_articles.append({
            "article_id": f"bench_{i}",
            "title": f"Benchmark Test Article {i} - Technology News",
            "description": f"This is a test description for benchmark article {i} about important technology developments.",
            "category": random.choice(categories),
            "source": random.choice(sources),
            "published_at": datetime.now(timezone.utc)
        })
    
    # Benchmark embedding generation
    print("1. Embedding Generation:")
    start = time.time()
    for article in test_articles:
        embedding = transforms._generate_embedding(article["title"], article["description"])
    embedding_time = time.time() - start
    
    print(f"   Time: {embedding_time:.3f}s")
    print(f"   Throughput: {n_articles / embedding_time:.0f} articles/second")
    
    # Benchmark clustering
    print("\n2. Clustering Assignment:")
    from streaming.tables import pathway_tables
    pathway_tables.initialize()
    
    start = time.time()
    for article in test_articles:
        embedding = transforms._generate_embedding(article["title"], article["description"])
        article["embedding"] = embedding
        pathway_tables._process_clustering(article)
    clustering_time = time.time() - start
    
    print(f"   Time: {clustering_time:.3f}s")
    print(f"   Throughput: {n_articles / clustering_time:.0f} articles/second")
    
    # Benchmark keyword extraction
    print("\n3. Keyword Extraction:")
    from services.event_clustering import clustering_service
    
    start = time.time()
    for article in test_articles:
        keywords = clustering_service._extract_keywords(article["title"] + " " + article["description"])
    keyword_time = time.time() - start
    
    print(f"   Time: {keyword_time:.3f}s")
    print(f"   Throughput: {n_articles / keyword_time:.0f} articles/second")
    
    # Summary
    total_time = embedding_time + clustering_time + keyword_time
    print(f"\n=== Total Time: {total_time:.3f}s ===")
    print(f"Overall throughput: {n_articles / total_time:.0f} articles/second")
    
    return {
        "embedding_time": embedding_time,
        "clustering_time": clustering_time,
        "keyword_time": keyword_time,
        "total_time": total_time,
        "throughput": n_articles / total_time
    }


def benchmark_api_response(num_requests: int = 100):
    """Benchmark API response time"""
    import httpx
    
    print(f"\n=== API Response Benchmark ({num_requests} requests) ===\n")
    
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/health",
        "/api/v1/events",
        "/api/v1/stats"
    ]
    
    results = {}
    
    for endpoint in endpoints:
        print(f"Testing endpoint: {endpoint}")
        
        times = []
        for i in range(num_requests):
            try:
                start = time.time()
                response = httpx.get(f"{base_url}{endpoint}", timeout=5)
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception as e:
                print(f"   Error: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results[endpoint] = {
                "avg": avg_time,
                "min": min_time,
                "max": max_time,
                "requests": len(times)
            }
            
            print(f"   Avg: {avg_time*1000:.1f}ms")
            print(f"   Min: {min_time*1000:.1f}ms")
            print(f"   Max: {max_time*1000:.1f}ms")
    
    return results


def benchmark_clustering_similarity(n_articles: int = 50, n_clusters: int = 10):
    """Benchmark similarity computation for clustering"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n=== Clustering Similarity Benchmark ===")
    print(f"Articles: {n_articles}, Existing clusters: {n_clusters}\n")
    
    # Generate random embeddings
    np.random.seed(42)
    article_embeddings = np.random.rand(n_articles, 384)
    cluster_centers = np.random.rand(n_clusters, 384)
    
    # Benchmark pairwise similarity
    print("1. Article-to-Cluster Similarity:")
    start = time.time()
    similarities = cosine_similarity(article_embeddings, cluster_centers)
    sim_time = time.time() - start
    
    print(f"   Time: {sim_time*1000:.2f}ms")
    print(f"   Matrix shape: {similarities.shape}")
    print(f"   Throughput: {n_articles * n_clusters / sim_time:.0f} comparisons/second")
    
    # Benchmark finding best match
    print("\n2. Best Match Finding:")
    start = time.time()
    for i in range(n_articles):
        best_cluster = np.argmax(similarities[i])
        best_score = similarities[i][best_cluster]
    match_time = time.time() - start
    
    print(f"   Time: {match_time*1000:.2f}ms")
    print(f"   Throughput: {n_articles / match_time:.0f} articles/second")
    
    return {
        "similarity_time": sim_time,
        "match_time": match_time
    }


def run_all_benchmarks():
    """Run all benchmarks"""
    print("=" * 60)
    print("LiveBrief Performance Benchmarks")
    print("=" * 60)
    
    # Run article processing benchmark
    article_results = benchmark_article_processing(n_articles=100)
    
    # Run clustering similarity benchmark
    similarity_results = benchmark_clustering_similarity(
        n_articles=50, n_clusters=10
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Article processing: {article_results['throughput']:.0f} articles/second")
    print(f"Similarity computation: {similarity_results['similarity_time']*1000:.2f}ms for 500 comparisons")
    
    return {
        "article_processing": article_results,
        "similarity": similarity_results
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "api":
            benchmark_api_response(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
        elif sys.argv[1] == "similarity":
            benchmark_clustering_similarity(
                int(sys.argv[2]) if len(sys.argv) > 2 else 50,
                int(sys.argv[3]) if len(sys.argv) > 3 else 10
            )
        else:
            print("Usage: python benchmarks.py [api|similarity|all]")
    else:
        run_all_benchmarks()

