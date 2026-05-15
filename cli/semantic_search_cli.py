#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify the model for semantic search")
    
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a text using the model")
    embed_text_parser.add_argument("text", type=str, help="Text to get the embedding")
    
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a query using the model")
    embed_query_parser.add_argument("query", type=str, help="Text to get the embedding")

    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            query = args.query
            embed_text(query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()