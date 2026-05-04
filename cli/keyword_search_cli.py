import argparse
import json
from text_processing import tokenize, read_stopwords
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    subparsers.add_parser("build", help="Build Inverted Index")
    

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            with open("data/movies.json", 'r') as f:
                movies = json.load(f)
            
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