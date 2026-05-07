import os
import json
import math
from text_processing import tokenize, read_stopwords
from pickle import dump, load
from collections import Counter

class InvertedIndex:
    index: dict[str, set[int]] = {}
    docmap: dict[int, dict[str, str]] = {}
    term_frequencies: dict[int, Counter] = {}
    
    stopwords = read_stopwords()
    
    def __add_document(self, doc_id: int, text: str):
        
        tokens = tokenize(text, stopwords=self.stopwords)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set({doc_id})
            else:
                self.index[token].add(doc_id)
                
        self.term_frequencies[doc_id] = Counter(tokens)
                
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
    
    def build(self):
        with open("data/movies.json", 'r') as f:
            movies = json.load(f)
            
        for movie in movies["movies"]:
            movie_dict = {
                "id": movie["id"],
                "title": movie["title"],
                "description": movie["description"]
            }
            
            doc_id = int(movie["id"])
            
            self.docmap[doc_id] = movie_dict
            
            self.__add_document(doc_id, text=f"{movie['title']} {movie['description']}" )
    
    def save(self):
        dest_dir = os.path.abspath("cache")
        os.makedirs(dest_dir, exist_ok=True)
        with open(os.path.join("/", *[dest_dir, "index.pkl"]), "+w") as f:
            dump(self.index, f.buffer)
            
        with open(os.path.join("/", *[dest_dir, "docmap.pkl"]), "+w") as f:
            dump(self.docmap, f.buffer)
            
        with open(os.path.join("/", *[dest_dir, "term_frequencies.pkl"]), "+w") as f:
            dump(self.term_frequencies, f.buffer)
            
    def load(self):
        src_dir = os.path.abspath("cache")
        index_file_path = os.path.join("/", *[src_dir, "index.pkl"])
        
        with open(index_file_path, "+rb") as f:
            self.index = load(f)
        
        docmap_file_path = os.path.join("/", *[src_dir, "docmap.pkl"])
        with open(docmap_file_path, "+rb") as f:
            self.docmap = load(f)
            
        term_frequencies_path = os.path.join("/", *[src_dir, "term_frequencies.pkl"])
        with open(term_frequencies_path, "+rb") as f:
            self.term_frequencies = load(f)
            
            