import os
from sentence_transformers import SentenceTransformer
import numpy as np
from search_utils import CACHE_DIR, load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_SIZE


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
    
    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embds = self.generate_embedding(query)
        similarity_score_doc = []
        
        for doc_idx in range(0,len(self.documents)):
            doc = self.documents[doc_idx]
            doc_embd = self.embeddings[doc_idx]
            
            similarity_score = cosine_similarity(query_embds, doc_embd)
            similarity_score_doc.append((similarity_score, doc))
        
        similarity_score_doc = sorted(similarity_score_doc, key=lambda x: x[0], reverse=True)
        
        top_results_limit = min(len(similarity_score_doc), limit)
        top_results = []
        
        for top_idx in range(0, top_results_limit):
            score, doc = similarity_score_doc[top_idx]
            top_results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"]
                }
            )
            
        return top_results
        
        
        
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
    
def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    semantic_search = SemanticSearch()
    
    documents = load_movies()
    
    semantic_search.load_or_create_embeddings(documents)
    
    results = semantic_search.search(query, limit)
    
    for result_idx in range(0,len(results)):
        result = results[result_idx]
        print(f"{result_idx+1}. {result["title"]} (score: {result['score']:.4f})")
        print(f"\t{result["description"]}")
        print("")

def chunk_command(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP_SIZE ):
    print(f"Chunking {len(text)} characters")

    chunks = split_text_in_chunks(text, chunk_size, overlap)
    print_chunks(chunks)

def split_text_in_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    
    num_full_chunks = len(words) // chunk_size
    remainder = len(words) % chunk_size

    first_idx = 0
    last_idx = chunk_size
    for _ in range(0, num_full_chunks):
        chunk = " ".join(words[first_idx:last_idx])
        chunks.append(chunk)
        first_idx = last_idx - overlap
        last_idx += chunk_size

    if(remainder > 0):
        last_idx = first_idx + remainder + overlap
        chunk = " ".join(words[first_idx:last_idx])
        chunks.append(chunk)
    
    return chunks


def print_chunks(chunks: list[str]):
    for chunk_idx in range(0, len(chunks)):
        print(f"{chunk_idx+1}. {chunks[chunk_idx]}")


    
        

        