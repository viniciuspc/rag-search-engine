import string

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = remove_punctioation(processed_text)
    
    return processed_text

def remove_punctioation(text):
    t = str.maketrans("", "", string.punctuation)
    
    return text.translate(t)