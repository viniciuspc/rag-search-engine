import os
import time
import json

from inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    RRF_K,
    LIMIT_MULTIPLIER,
    load_movies,
)
from .individual_re_ranker import calculate_rank
from .batch_re_ranker import calculate_batch_rank
from sentence_transformers import CrossEncoder
from log_utils import log_results
from lib.rrf_evaluator import evaluate_rrf
from lib.rag_agent import run_rag

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
        
        keywords_scores: list[float] = []
        
        for _, keyword_score in keyword_results:
            keywords_scores.append(keyword_score)
            
        normalized_keyword_socores = normalize(keywords_scores)
        
        semantic_scores: list[float] = []
        for semantic_result in semantic_results:
            semantic_scores.append(semantic_result["score"])
        
        normalized_semantic_scores = normalize(semantic_scores)
        
        doc_scores = {}
        idx_scores = 0
        
        for doc_id, keyword_score in keyword_results:
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {}
                doc_scores[doc_id]["document"] = self.document_map[doc_id]
            doc_scores[doc_id]["keyword_score"] = normalized_keyword_socores[idx_scores]
            idx_scores += 1
        
        idx_scores = 0
        for semantic_result in semantic_results:
            doc_id = semantic_result["id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {}
                doc_scores[doc_id]["document"] = self.document_map[doc_id]
            doc_scores[doc_id]["semantic_score"] = normalized_semantic_scores[idx_scores]
            idx_scores += 1
            
        for doc_id in doc_scores.keys():
            doc_score = doc_scores[doc_id]
            keyword_score = 0.0
            
            if "keyword_score" in doc_score: 
                keyword_score = doc_score["keyword_score"]
            
            semantic_score = 0.0
            if "semantic_score" in doc_score:
                semantic_score = doc_score["semantic_score"]
            
            doc_scores[doc_id]["hybrid_score"] = hybrid_score(
                bm25_score=keyword_score, 
                semantic_score=semantic_score, 
                alpha=alpha
            )
            
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
        limit_with_buffer = limit * 500
        keyword_results = self._bm25_search(query, limit_with_buffer)
        semantic_results = self.semantic_search.search_chunks(query, limit_with_buffer)
        
        doc_ranks = {}
        
        idx_scores = 0
        
        for doc_id, keyword_score in keyword_results:
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {}
                doc_ranks[doc_id]["document"] = self.document_map[doc_id]
            doc_ranks[doc_id]["keyword_rank"] = idx_scores + 1
            idx_scores += 1
            
        idx_scores = 0
        for semantic_result in semantic_results:
            doc_id = semantic_result["id"]
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {}
                doc_ranks[doc_id]["document"] = self.document_map[doc_id]
            doc_ranks[doc_id]["semantic_rank"] = idx_scores + 1
            idx_scores += 1
            
        for doc_id in doc_ranks.keys():
            doc_score = doc_ranks[doc_id]
            keyword_score = 0.0
            
            if "keyword_rank" in doc_score: 
                keyword_score = rrf_score(doc_score["keyword_rank"], k)
            
            semantic_score = 0.0
            if "semantic_rank" in doc_score:
                semantic_score = rrf_score(doc_score["semantic_rank"], k)
            
            doc_ranks[doc_id]["rrf_score"] = keyword_score + semantic_score
            
        doc_scores = dict(
            sorted(
                doc_ranks.items(), key=lambda x: x[1]["rrf_score"],
                reverse=True
            ))
        
        top_results: list[dict] = [scores for _, scores in list(doc_scores.items())[:limit]]
        
        log_results("RRF Serach results", top_results)
            
        return top_results
    
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

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

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
        
        
def rrf_search_command(query: str, k: int = RRF_K, limit = DEFAULT_SEARCH_LIMIT, evaluate: bool = False):
    movies = load_movies()
    
    hybrid_search = HybridSearch(documents=movies)
    
    results = hybrid_search.rrf_search(
        query,
        k,
        limit
    )
    
    for i, result in enumerate(results):
        bm25_rank = format_rank(result, "keyword_rank")
        
        semantinc_rank = format_rank(result, "semantic_rank")
        
        print(f"\n{i+1}. {result["document"]["title"]}")
        print(f"   RRF Score: {result['rrf_score']:.4f}")
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantinc_rank}")
        print(f"   {result['document']['description'][:100]}...")
        
    if evaluate:
        evaluate_rrf_with_llm(query, results)
        
def rrf_search_rehank_command(query: str, k: int = RRF_K, limit = DEFAULT_SEARCH_LIMIT):
    print(f"Re-ranking top {limit} results using individual method...")
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    
    sleep_time_in_seconds = 1
    
    movies = load_movies()
    
    hybrid_search = HybridSearch(documents=movies)
    
    
    results = hybrid_search.rrf_search(
        query,
        k,
        limit * LIMIT_MULTIPLIER
    )
    
    calculating_index = 1
    
    for result in results:
        print(f"Calculating: {calculating_index} of {len(results)}")
        llm_rank = calculate_rank(query, result["document"])
        print(f"Rank {llm_rank} with BM25 Rank {result["keyword_rank"]} calculated for {result["document"]["title"]}")
        result["re_rank"] = llm_rank
        time.sleep(sleep_time_in_seconds)
        calculating_index+=1
    
        
    results = sorted(
                results, key=lambda x: x["re_rank"],
                reverse=True
            )[:limit]
    
    for i, result in enumerate(results):
        bm25_rank = format_rank(result, "keyword_rank")
        
        semantinc_rank = format_rank(result, "semantic_rank")
        
        print(f"\n{i+1}. {result["document"]["title"]}")
        print(f"   Re-rank Score: {result['re_rank']:.3f}/10")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantinc_rank}")
        print(f"   {result['document']['description'][:100]}...")
        

def rrf_search_rehank_batch_command(query: str, k: int = RRF_K, limit = DEFAULT_SEARCH_LIMIT):
    print(f"Re-ranking top {limit} results using batch method...")
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    
    movies = load_movies()
    
    hybrid_search = HybridSearch(documents=movies)
    
    RE_RANK_FOR_UNKNOW_ID = 999
    
    rrf_search_results = hybrid_search.rrf_search(
        query,
        k,
        limit * LIMIT_MULTIPLIER
    )
    
    doc_list_str = json.dumps(rrf_search_results)
    
    llm_ranks = calculate_batch_rank(query, doc_list_str)
    
    for rrf_search_result in rrf_search_results:
        id = rrf_search_result["document"]["id"]
        if(id in llm_ranks):
            rrf_search_result["re_rank"] = llm_ranks.index(id) + 1
        else:
            rrf_search_result["re_rank"] = RE_RANK_FOR_UNKNOW_ID
        
    
    
    
    results = sorted(
                rrf_search_results, key=lambda x: x["re_rank"],
                reverse=False
            )[:limit]
    
    for i, result in enumerate(results):
        bm25_rank = format_rank(result, "keyword_rank")
        
        semantinc_rank = format_rank(result, "semantic_rank")
        
        print(f"\n{i+1}. {result["document"]["title"]}")
        print(f"   Re-rank Rank: {result['re_rank']}")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantinc_rank}")
        print(f"   {result['document']['description'][:100]}...")
        
def rrf_search_rehank_cross_encoder_command(query: str, k: int = RRF_K, limit = DEFAULT_SEARCH_LIMIT):
    print(f"Re-ranking top {limit} results using cross_encoder method...")
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    
    movies = load_movies()
    
    hybrid_search = HybridSearch(documents=movies)
    
    rrf_search_results = hybrid_search.rrf_search(
        query,
        k,
        limit * LIMIT_MULTIPLIER
    )
    
    pairs = []
    
    for rrf_search_result in rrf_search_results:
            doc = rrf_search_result["document"]
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
            
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    
    # `predict` returns a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    
    for i, score in enumerate(scores):
        rrf_search_result = rrf_search_results[i]
        rrf_search_result["re_rank"] = score
        rrf_search_results[i] = rrf_search_result
    
    log_results("Results after rerank", rrf_search_results)
    
    results = sorted(
                rrf_search_results, key=lambda x: x["re_rank"],
                reverse=True
            )[:limit]
    
    for i, result in enumerate(results):
        bm25_rank = format_rank(result, "keyword_rank")
        
        semantinc_rank = format_rank(result, "semantic_rank")
        
        print(f"\n{i+1}. {result["document"]["title"]}")
        print(f"   Cross Encoder Score:: {result['re_rank']:.3f}")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantinc_rank}")
        print(f"   {result['document']['description'][:100]}...")
        
def rag_command(query: str):
    movies = load_movies()
        
    hybrid_search = HybridSearch(documents=movies)
    
    rrf_search_results = hybrid_search.rrf_search(
        query,
        RRF_K,
        DEFAULT_SEARCH_LIMIT
    )
    
    formatted_results = format_results(rrf_search_results)
    
    rag_response = run_rag(query, formatted_results)
    
    print("Search Results:")
    for result in rrf_search_results:
        title = result["document"]["title"]
        print(f"- {title}")
        
    print("RAG Response:")
    print(rag_response)
    
        
def format_rank(dictionary, key) -> str:
    rank = "-"
    if key in dictionary:
        rank = f"{dictionary[key]}"
        
    return rank

def evaluate_rrf_with_llm(query: str, results: list[dict]):
    formatted_results = format_results(results)
        
    rrf_evaluations: list[int] = evaluate_rrf(query, formatted_results)
    print_rrf_evaluation(results, rrf_evaluations)
    
def format_results(results: list[dict]) -> list[str]:
    formatted_results: list[str] = []
    for result in results:
        doc = result["document"]
        formatted_results.append(f"{doc["title"]}: {doc["description"]}")
    
    return formatted_results
    
def print_rrf_evaluation(results: list[dict], rrf_evaluations: list[int]):
    if len(results) != len(rrf_evaluations):
        raise ValueError("Results and rrf_evaluations length does not match")
    
    for i in range(0, len(results)):
        title = results[i]["document"]["title"]
        evaluation = rrf_evaluations[i]
        
        print(f"{i+1}. {title}: {evaluation}/3")
    
    
    