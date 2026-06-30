import argparse
from evaluation_utils import load_golden_dataset
from lib.hybrid_search import HybridSearch
from search_utils import load_movies, RRF_K

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    
    print(f"k={limit}")
    
    movies = load_movies()
    hybrid_search = HybridSearch(documents=movies)
    
    golden_dataset = load_golden_dataset()
    for test_case in golden_dataset:
        query: str = test_case["query"]
        
        print(f"- Query: {query}")
        
        relevant_docs: list[str] = test_case["relevant_docs"]
        
        retrieved_docs = hybrid_search.rrf_search(query=query, k=RRF_K, limit=limit)
        retrieved_titles = []
        relevant_retrieved = 0
        
        for retrieved_doc in retrieved_docs:
            title = retrieved_doc["document"]["title"]
            retrieved_titles.append(title)
            if title in relevant_docs:
                relevant_retrieved += 1
        
        total_retrieved = len(retrieved_docs)
        precision = relevant_retrieved / total_retrieved
        
        total_relevant = len(relevant_docs)
        recall = relevant_retrieved / total_relevant
        
        
        
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - Retrieved: {format_titles(retrieved_titles)}")
        print(f"    - Relevant: {format_titles(relevant_docs)}")
        
                
        
def format_titles(titles : list[str]) -> str:
    return ', '.join(titles)


if __name__ == "__main__":
    main()