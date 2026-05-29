import os

from inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    load_movies,
)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit).items()

    def weighted_search(self, query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        limit_with_buffer = limit * 500
        keyword_results = self._bm25_search(query, limit_with_buffer)
        semantic_results = self.semantic_search.search_chunks(query, limit_with_buffer)
        
        scores: list[float] = []
        
        for _, keyword_score in keyword_results:
            scores.append(keyword_score)
            
        for semantic_result in semantic_results:
            scores.append(semantic_result["score"])
            
        normalized_scores = normalize(scores)
        
        doc_scores = {}
        idx_scores = 0
        
        for doc_id, keyword_score in keyword_results:
            if doc_id not in doc_scores:
                doc_scores[doc_id]["document"] = self.document_map[doc_id]
            doc_scores[doc_id]["keyword_score"] = normalized_scores[idx_scores]
            idx_scores += 1
            
        for semantic_result in semantic_results:
            doc_id = semantic_result["id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id]["document"] = self.document_map[doc_id]
            doc_scores[doc_id]["semantic_score"] = normalized_scores[idx_scores]
            idx_scores += 1
            
        for doc_id in doc_scores.keys():
            keyword_score = doc_scores[doc_id]["keyword_score"]
            semantic_score = doc_scores[doc_id]["semantic_score"]
            doc_scores[doc_id]["hybrid_score"] = hybrid_score(bm25_score=keyword_score, semantic_score=semantic_score, alpha=alpha)
            
        doc_scores = dict(
            sorted(
                doc_scores.items(), key=lambda x: x[1]["hybrid_score"],
                reverse=False
            ))
        
        top_results_limit = min(len(doc_scores), limit)
        top_results: list[dict] = []
        
        for _ in range(0, top_results_limit):
            doc_id, scores = doc_scores.popitem()
            top_results.append(scores)
            
        return top_results
        
                
            

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize_command(scores: list[float]):
    if len(scores) == 0:
        return
    
    normalized_values = normalize(scores)
            
    for normalized_value in normalized_values:
        print(f"* {normalized_value:.4f}")
        
def normalize(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    
    normalized_values = []
    
    can_normalize = min_score != max_score
    
    for score in scores:
        normalized_score = 1.0
        if can_normalize:
            normalized_score = (score - min_score) / (max_score - min_score)
        
        normalized_values.append(normalized_score)
            
    return normalized_values

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    
    hybrid_search = HybridSearch(documents=movies)
    
    results = hybrid_search.weighted_search(
        query,
        alpha,
        limit
    )
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result["document"]["title"]}")
        print(f"   Hybrid Score:: {result['hybrid_score']:.4f}")
        print(f"   BM25: {result['keyword_score']:.4f}, Semantic: {result['semantic_score']:.4f}")
        print(f"   {result['document']['description'][:100]}...")