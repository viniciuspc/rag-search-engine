import os
import json
from lib.semantic_search import SemanticSearch, semantic_split_text_in_chunks
import numpy as np
from search_utils import CACHE_DIR, load_movies

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
            doc_chunks = semantic_split_text_in_chunks(description, max_chunk_size=4, overlap=1)
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
            self.chunk_metadata_path = json.load(f)
        
def embed_chunks_command():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    embeddings = chunked_semantic_search.chunk_embeddings
    
    print(f"Generated {len(embeddings)} chunked embeddings")
    
            
            
            
            
            
        
        
        