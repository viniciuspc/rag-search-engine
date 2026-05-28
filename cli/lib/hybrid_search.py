import os

from inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit).items()

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

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