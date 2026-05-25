import json
import os

import numpy as np
from lib.semantic_search import SemanticSearch, semantic_chunk, cosine_similarity
from search_utils import (
    CACHE_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    SearchResult,
    load_movies,
    format_search_result
)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        
        chunks: list[str] = []
        chunks_medata: list[dict] = []
        
        self.documents = documents
        
        for movie_idx in range(0, len(documents)):
            doc = documents[movie_idx]
            description = doc['description']
            self.document_map[doc["id"]] = doc
            
            if not description:
                continue
            doc_chunks = semantic_chunk(
                description, 
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, 
                overlap=DEFAULT_CHUNK_OVERLAP
            )
            chunks.extend(doc_chunks)
            
            for chunk_idx in range(0, len(doc_chunks)):
                chunks_medata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(doc_chunks)
                })
                
        encoded_chunks = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_embeddings = encoded_chunks
        self.chunk_metadata = chunks_medata
        
        self.save(len(chunks))
        
        return self.chunk_embeddings
        
    def save(self, total_chunks: int):
        dest_dir = CACHE_DIR
        os.makedirs(dest_dir, exist_ok=True)
        
        if self.chunk_embeddings is not None:
            with open(self.chunk_embeddings_path, "+w") as f:
                np.save(f.buffer, self.chunk_embeddings)
                
        if self.chunk_metadata is not None:
            with open(self.chunk_metadata_path, "+w") as f:
                json.dump(
                    {
                        "chunks": self.chunk_metadata, 
                        "total_chunks": total_chunks
                    }, 
                    f, indent=2)
                
    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if self.checkPathsExists():
            self.loadFiles()
        
        if self.chunk_embeddings is not None:
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
        
            
    def checkPathsExists(self):
        return os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path)
    
    def loadFiles(self):
        self.chunk_embeddings = np.load(self.chunk_embeddings_path)
        
        with open(self.chunk_metadata_path, "r") as f:
            self.chunk_metadata = json.load(f)
            
    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        if self.checkIfEmbeddingsAreNotLoaded():
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        query_embd = self.generate_embedding(query)
        chunk_scores = []
        
        for idx, chunk_embd in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata["chunks"][idx]
            similarity_score = cosine_similarity(query_embd, chunk_embd)
            
            chunk_scores.append({
                    "movie_idx": metadata["movie_idx"],
                    "chunk_idx": metadata["chunk_idx"],
                    "total_chunks": metadata["total_chunks"],
                    "score": similarity_score
                })
            
        movie_to_score = {}
        
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            should_update_score = (
                movie_idx not in movie_to_score or 
                movie_to_score[movie_idx]["score"] < chunk_score["score"]
            )
            if should_update_score:
                movie_to_score[movie_idx] = chunk_score
                
        movie_to_score = dict(sorted(movie_to_score.items(), key=lambda x: x[1]["score"], reverse=False))
        
        top_results_limit = min(len(movie_to_score), limit)
        top_results: list[SearchResult] = []
        
        for _ in range(0, top_results_limit):
            doc_id, chunk_score = movie_to_score.popitem()
            print(chunk_score)
            doc = self.documents[doc_id]
            search_result = format_search_result(
                doc_id=doc_id,
                title=doc["title"],
                document=doc["description"][:100],
                score=chunk_score["score"],
                metadata={
                    "chunk_idx": chunk_score["chunk_idx"], 
                    "total_chunks": chunk_score["total_chunks"]
                }
            )
            top_results.append(search_result)
        
        return top_results
        
                

        
    
    def checkIfEmbeddingsAreNotLoaded(self) -> bool:
        return self.chunk_embeddings is None or self.documents is None or self.chunk_metadata is None
        
        
def embed_chunks_command():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    embeddings = chunked_semantic_search.chunk_embeddings
    
    if embeddings is not None:
        print(f"Generated {len(embeddings)} chunked embeddings")
        
def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    
    results = chunked_semantic_search.search_chunks(query, limit)
    
    for i, result in enumerate(results):
        print(f"\n{i}. {result["title"]} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
    
            
            
            
            
            
        
        
        