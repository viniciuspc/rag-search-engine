import argparse
from lib.multimodal_search import verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Print a image embedding for the given image"
    )
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Image path to generate the enconding")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            verify_image_embedding(image_path)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()