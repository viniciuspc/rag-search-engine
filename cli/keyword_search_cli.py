import argparse
import json
from text_processing import preprocess_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            with open("data/movies.json", 'r') as f:
                movies = json.load(f)
                
            results_list = []
            
            for movie in movies["movies"]:
                title = movie["title"]
                if preprocess_text(query) in preprocess_text(title):
                    results_list.append(title)
            
            if len(results_list) == 0:
                print("No result found.")
                return
            
            for i in range(min(len(results_list), 5)):
                print(f"{i+1}. {results_list[i]}")
            
        case _:
            parser.print_help()
            


if __name__ == "__main__":
    main()