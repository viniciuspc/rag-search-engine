import string
from nltk.stem import PorterStemmer

def preprocess_text(text) -> str:
    processed_text = text.lower()
    processed_text = remove_punctioation(processed_text)
    
    return processed_text

def remove_punctioation(text: str) -> str:
    t = str.maketrans("", "", string.punctuation)
    
    return text.translate(t)

def tokenize(text: str, stopwords: list[str]) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    fileterd_tokens = []
    for token in tokens:
        if should_add_token(token, stopwords):
            fileterd_tokens.append(stem_token(token))
            
    return fileterd_tokens

def should_add_token(token: str, stopwords) -> bool:
    if len(token) == 0:
        return False
    
    for stopword in stopwords:
        
        if token == stopword:
            return False
        
    return True

def stem_token(token: str) -> str:
    stemmer = PorterStemmer()
    return stemmer.stem(token)

def read_stopwords() -> list[str]:
    with open("data/stopwords.txt", 'r') as f:
                file_content = f.read()
                
    return file_content.splitlines()