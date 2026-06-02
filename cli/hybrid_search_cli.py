import argparse

from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command
from search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    RRF_K
)
from lib.spell_corrector import correct_spell
from lib.query_rewriter import rewrite_query
from lib.query_expander import expand_query

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", nargs='*', help="List of scores to normalize")

    wighted_search_parser = subparsers.add_parser("weighted-search", help="Search using mixed keyoard and semantic serachs")
    wighted_search_parser.add_argument("query", type=str, help="Text to search")
    wighted_search_parser.add_argument(
        "--alpha", 
        type=float, 
        nargs='?', 
        default=DEFAULT_ALPHA, 
        help="Weight between keyword and semantic search 1.0 100% Keyword. 0.0 100% Semantic", 
        required=False
    )
    wighted_search_parser.add_argument(
        "--limit", 
        type=int, 
        nargs='?', 
        default=DEFAULT_SEARCH_LIMIT, 
        help="Number of documents to return", 
        required=False
    )
    
    rff_search_parser = subparsers.add_parser(
        "rrf-search", 
        help="Search using Reciprocal Ranking Funsion"
    )
    rff_search_parser.add_argument("query", type=str, help="Text to search")
    rff_search_parser.add_argument(
        "-k", 
        type=float, 
        nargs='?', 
        default=RRF_K, 
        help="Weight between keyword and semantic search 1.0 100% Keyword. 0.0 100% Semantic", 
        required=False
    )
    rff_search_parser.add_argument(
        "--limit", 
        type=int, 
        nargs='?', 
        default=DEFAULT_SEARCH_LIMIT, 
        help="Number of documents to return", 
        required=False
    )
    rff_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            scores = [float(i) for i in scores]
            normalize_command(scores)
        case "weighted-search":
            query = args.query
            alpha = args.alpha
            limit = args.limit
            weighted_search_command(query, alpha, limit)
        case "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit
            
            method = args.enhance
            
            enhanced_query = query
            if method == "spell":
                enhanced_query = correct_spell(query)
                print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
            elif method == "rewrite":
                enhanced_query = rewrite_query(query)
                print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
            elif method == "expand":
                enhanced_query = expand_query(query)
                print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
                
            
            rrf_search_command(enhanced_query, k, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()