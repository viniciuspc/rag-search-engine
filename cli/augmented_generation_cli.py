import argparse
from lib.hybrid_search import rag_command, summarize_command
from search_utils import DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform summarization in the search result"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", 
        type=int, 
        nargs='?', 
        default=DEFAULT_SEARCH_LIMIT, 
        help="Number of documents to return", 
        required=False
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)
        case "summarize":
            query = args.query
            limit = args.limit
            
            summarize_command(query, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()