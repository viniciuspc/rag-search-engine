import argparse
from text_processing import tokenize, read_stopwords
from inverted_index import InvertedIndex, bm25_tf_command
from constants import BM25_K1

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
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for document id and term")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID to get the TF-IDF")
    tfidf_parser.add_argument("term", type=str, help="Term to get the TF-IDF")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    bm25_tf_parser = subparsers.add_parser( "bm25tf", help="Get BM25 TF score for a given document ID and term")
    
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

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
            
        case "tfidf":
            doc_id = int(args.doc_id)
            term = args.term
            
            invertded_index = InvertedIndex()
            
            try:
                invertded_index.load()
            except FileNotFoundError:
                print("Index not created yet. Run build first.")
                return
            
            tf_idf = invertded_index.get_tfidf(doc_id, term)
            
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            
        case "bm25idf":
            
            term = args.term
            
            invertded_index = InvertedIndex()
            
            try:
                invertded_index.load()
            except FileNotFoundError:
                print("Index not created yet. Run build first.")
                return
            
            bm25idf = invertded_index.get_bm25_idf(term)
            
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            doc_id = int(args.doc_id)
            term = args.term
            k1 = args.k1
            
            bm25tf = bm25_tf_command(doc_id, term, k1)
            
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            
            
        
            
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