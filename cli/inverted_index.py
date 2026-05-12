import os
import math
from text_processing import tokenize
from pickle import dump, load
from collections import Counter
from search_utils import BM25_B, BM25_K1, CACHE_DIR, load_movies, load_stopwords

class InvertedIndex:
    index: dict[str, set[int]] = {}
    docmap: dict[int, dict[str, str]] = {}
    term_frequencies: dict[int, Counter] = {}
    doc_lengths: dict[int, int] = {}
    
    index_path = os.path.join(CACHE_DIR, "index.pkl")
    docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
    term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
    doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl") 
    
    stopwords = load_stopwords()
    
    def __add_document(self, doc_id: int, text: str):
        
        tokens = tokenize(text, stopwords=self.stopwords)
        
        self.doc_lengths[doc_id] = len(tokens)
        
        for token in tokens:
            if token not in self.index:
                self.index[token] = set({doc_id})
            else:
                self.index[token].add(doc_id)
                
        self.term_frequencies[doc_id] = Counter(tokens)
        
    def __get_avg_doc_length(self) -> float:
        if(len(self.doc_lengths) == 0):
            return 0
        
        sum_doc_length = 0
        
        for doc_lengh in self.doc_lengths.values():
            sum_doc_length += doc_lengh
            
        return sum_doc_length / len(self.doc_lengths)
        
                
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term, self.stopwords)
        if len(tokens) > 1:
            raise Exception("Term has to much tokens. Make sure the torm have only one token.")
        
        if doc_id not in self.term_frequencies:
            raise Exception(f"doc_id {doc_id} not found in term frequencies {self.term_frequencies}")
        
        tf = self.term_frequencies[doc_id][tokens[0]]
        return tf
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize(term, self.stopwords)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term, self.stopwords)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        
        bm25_idf = math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
        
        return bm25_idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
        
    
    def build(self):
        movies = load_movies()
            
        for movie in movies:
            movie_dict = {
                "id": movie["id"],
                "title": movie["title"],
                "description": movie["description"]
            }
            
            doc_id = int(movie["id"])
            
            self.docmap[doc_id] = movie_dict
            
            self.__add_document(doc_id, text=f"{movie['title']} {movie['description']}" )
    
    def save(self):
        dest_dir = CACHE_DIR
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Saving cache to: {CACHE_DIR}")
        with open(self.index_path, "+w") as f:
            dump(self.index, f.buffer)
            
        with open(self.docmap_path, "+w") as f:
            dump(self.docmap, f.buffer)
            
        with open(self.term_frequencies_path, "+w") as f:
            dump(self.term_frequencies, f.buffer)
            
        with open(self.doc_lengths_path, "w+") as f:
            dump(self.doc_lengths, f.buffer)
            
    def load(self):
        
        with open(self.index_path, "+rb") as f:
            self.index = load(f)
        
        with open(self.docmap_path, "+rb") as f:
            self.docmap = load(f)
            
        with open(self.term_frequencies_path, "+rb") as f:
            self.term_frequencies = load(f)
        
        with open(self.doc_lengths_path, "+rb") as f:
            self.doc_lengths = load(f)
            
            
def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    invertded_index = InvertedIndex()
            
    try:
        invertded_index.load()
    except FileNotFoundError:
        print("Index not created yet. Run build first.")
        return
    
    return invertded_index.get_bm25_tf(doc_id, term, k1, b)
            
            
            