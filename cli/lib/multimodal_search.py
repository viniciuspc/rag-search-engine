from PIL import Image
from sentence_transformers import SentenceTransformer
from search_utils import load_movies, texts_from_documents
from lib.cosine_similarity_calculator import cosine_similarity

class MultimodalSearch():
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = texts_from_documents(documents)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        
    def embed_image(self, image_path: str):
        image_file = Image.open(image_path)
        embeddings = self.model.encode(image_file)
        return embeddings
    
    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)
        doc_scores: list[dict] = []
        
        for idx, text_embd in enumerate(self.text_embeddings):
            doc = self.documents[idx]
            similarity_score = cosine_similarity(image_embedding, text_embd)
            doc_scores.append({
                "doc_id": doc["id"],
                "doc_title": doc["title"],
                "doc_description": doc["description"],
                "similarity_score": similarity_score
            })
            
        doc_scores = sorted(
                doc_scores, key=lambda x: x["similarity_score"],
                reverse=True
            )
        
        top_results: list[dict] = doc_scores[:5]
        
        return top_results
            
        
        
    
def verify_image_embedding(image_path: str):
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents)
    
    embedding = multimodal_search.embed_image(image_path)
    
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
    
def image_search_command(image_path: str):
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents)
    
    return multimodal_search.search_with_image(image_path)