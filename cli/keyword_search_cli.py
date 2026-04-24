import argparse
import json
from text_processing import tokenize, read_stopwords
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build Inverted Index")
    

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            with open("data/movies.json", 'r') as f:
                movies = json.load(f)
                
            results_list = []
            stopwords = read_stopwords()
            query_tokens = tokenize(query, stopwords)
            
            for movie in movies["movies"]:
                title = movie["title"]
                
                title_tokens = tokenize(title, stopwords)
                if has_matching_token(query_tokens, title_tokens):
                    results_list.append(title)
                
            if len(results_list) == 0:
                print("No result found.")
                return
            
            for i in range(min(len(results_list), 5)):
                print(f"{i+1}. {results_list[i]}")
        case "build":
            invertded_index = InvertedIndex()
            invertded_index.build()
            invertded_index.save()
            
            docs = invertded_index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
            
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