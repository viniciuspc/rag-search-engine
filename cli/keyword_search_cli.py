import argparse
import math
from text_processing import tokenize, read_stopwords
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    subparsers.add_parser("build", help="Build Inverted Index")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency from doc_id and term")
    tf_parser.add_argument("doc_id", type=int, help="Document to get the term frequency")
    tf_parser.add_argument("term", type=str, help="Term to get the frequency")
    
    idf_parser = subparsers.add_parser("idf", help="Get Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term to get the frequency")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            
            invertded_index = InvertedIndex()
            
            try:
                invertded_index.load()
            except FileNotFoundError:
                print("Index not created yet. Run build first.")
                return
                
            results_list = []
            stopwords = read_stopwords()
            query_tokens = tokenize(query, stopwords)
            
            for query_token in query_tokens:
                documents = invertded_index.get_documents(query_token)
                
                spaces_remaining = 5 - len(results_list)
                
                for document_id in documents[0:spaces_remaining]:
                    results_list.append(document_id)
                    
                if len(results_list) >= 5:
                    break
                
            if len(results_list) == 0:
                print("No result found.")
                return
            
            for document_id in results_list[0:5]:
                movie_title = invertded_index.docmap[document_id]["title"]
                print(f"{document_id}. {movie_title}")
        case "build":
            invertded_index = InvertedIndex()
            invertded_index.build()
            invertded_index.save()
        case "tf":
            doc_id = int(args.doc_id)
            term = args.term
            
            invertded_index = InvertedIndex()
            
            try:
                invertded_index.load()
            except FileNotFoundError:
                print("Index not created yet. Run build first.")
                return
            
            tf = invertded_index.get_tf(doc_id, term)
            print(tf)
            
        case "idf":
            term = args.term
            
            invertded_index = InvertedIndex()
            
            try:
                invertded_index.load()
            except FileNotFoundError:
                print("Index not created yet. Run build first.")
                return
            
            idf = invertded_index.get_idf(term)
            
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        
            
        case _:
            parser.print_help()
            
            
def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


if __name__ == "__main__":
    main()