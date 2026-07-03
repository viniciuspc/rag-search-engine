import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Print a image embedding for the given image"
    )
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Image path to generate the enconding")
    
    image_search_parser = subparsers.add_parser(
            "image_search", help="Search using an image"
        )
    image_search_parser.add_argument("image_path", type=str, help="Image to search movies")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            verify_image_embedding(image_path)
        case "image_search":
            image_path = args.image_path
            results = image_search_command(image_path)
            
            print_image_search_results(results)
            
            
        case _:
            parser.print_help()
            
def print_image_search_results(results: list[dict]):
    for idx, result in enumerate(results):
        print(f"{idx+1}. {result["doc_title"]} (similarity: {result["similarity_score"]:.3})")
        print(f"   {result["doc_description"][:100]}...")

if __name__ == "__main__":
    main()