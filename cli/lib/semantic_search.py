import os
from sentence_transformers import SentenceTransformer
import numpy as np
from search_utils import CACHE_DIR, load_movies


class SemanticSearch():
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        
    def generate_embedding(self, text: str):
        text = text.strip()
        
        if len(text) == 0:
            raise ValueError("Text is empty")
        
        embedding = self.model.encode([text])[0]
        return embedding
    
    def build_embeddings(self, documents: list):
        self.documents = documents
        
        list_str_movies: list[str] = []
        
        for doc in documents:
            self.document_map[doc["id"]] = doc
            list_str_movies.append(f"{doc['title']}: {doc['description']}")
            
        self.embeddings = self.model.encode(list_str_movies, show_progress_bar=True)
        
        self.save()
        
        return self.embeddings
        
    def save(self):
        dest_dir = CACHE_DIR
        os.makedirs(dest_dir, exist_ok=True)
        if self.embeddings is not None:
            with open(self.embeddings_path, "+w") as f:
                np.save(f.buffer, self.embeddings)
                
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        
        for doc in documents:
            self.document_map[doc["id"]] = doc
            
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        
        # Build or rebuild embedding
        return self.build_embeddings(documents)
        
            
        
        
            
        
            
        
def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
    
def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def verify_embeddings():
    semantic_search = SemanticSearch()
    
    documents = load_movies()
    
    semantic_search.load_or_create_embeddings(documents)
    
    embeddings = semantic_search.embeddings
    
    print(f"Number of docs:   {len(documents)}")
    if embeddings is not None:
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    else:
        print("No embeddings")
        
def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    
        