"""
Experiment: Test different top-k values on Part 1 subset
Compare retrieval quality with k=2, k=4 (current), and k=8
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import embed_texts
from config import WIKI_COLLECTION, CHROMA_PATH, TOP_K
import chromadb
import json

# Subset of Part 1 queries for testing
TEST_QUERIES = [
    "What is photosynthesis?",
    "What is a volcano?",
    "What is the Internet?",
    "Who was Albert Einstein?",
    "What is the Taj Mahal?",
]

def evaluate_top_k(k_value):
    """Evaluate retrieval at a specific top-k value"""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(WIKI_COLLECTION)
    
    results = []
    
    for query in TEST_QUERIES:
        # Embed query
        query_embedding = embed_texts([query])[0]
        
        # Query with specific k
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k_value,
            include=["documents", "metadatas", "distances"]
        )
        
        top_score = 1 - query_result["distances"][0][0] if query_result["distances"][0] else 0
        num_retrieved = len(query_result["documents"][0]) if query_result["documents"] else 0
        
        results.append({
            "query": query,
            "top_k": k_value,
            "num_retrieved": num_retrieved,
            "top_score": float(top_score),
            "top_title": query_result["metadatas"][0][0].get("title", "unknown") if query_result["metadatas"] and query_result["metadatas"][0] else "unknown"
        })
    
    return results

def main():
    print("Testing different top-k values...\n")
    
    all_results = []
    
    for k in [2, 4, 8]:
        print(f"Running with k={k}...")
        results = evaluate_top_k(k)
        all_results.extend(results)
        print(f"  Completed {len(results)} queries")
    
    # Aggregate statistics
    print("\n=== TOP-K COMPARISON RESULTS ===\n")
    for k in [2, 4, 8]:
        k_results = [r for r in all_results if r['top_k'] == k]
        avg_score = sum(r['top_score'] for r in k_results) / len(k_results) if k_results else 0
        print(f"k={k}:")
        print(f"  Average top score: {avg_score:.4f}")
        print(f"  All queries retrieved: {all(r['num_retrieved'] == k for r in k_results)}")
        for r in k_results:
            print(f"    '{r['query'][:40]}...' → top: {r['top_title']} (score: {r['top_score']:.4f})")
        print()
    
    # Save results
    with open('./artifacts/top_k_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to artifacts/top_k_comparison.json")

if __name__ == "__main__":
    main()
