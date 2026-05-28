import argparse

from lib.hybrid_search import normalize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", nargs='*', help="List of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            scores = [float(i) for i in scores]
            normalize_command(scores)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()