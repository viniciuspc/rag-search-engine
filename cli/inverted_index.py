import os

import json
from text_processing import tokenize, read_stopwords
from pickle import dump

class InvertedIndex:
    index: dict[str, set[int]] = {}
    docmap: dict[int, dict[str, str]] = {}
    
    stopwords = read_stopwords()
    
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize(text, stopwords=self.stopwords)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set({doc_id})
            else:
                self.index[token].add(doc_id)
                
    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index[term.lower()])
    
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
            
            