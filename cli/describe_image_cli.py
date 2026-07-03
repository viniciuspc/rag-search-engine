import argparse
from image_describer import describe_image_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe image CLI")
    parser.add_argument(
        "--image", 
        type=str, 
        help="A path to image that the cli will describe."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Search query"
    )
    

    args = parser.parse_args()
    
    image_path = args.image    
    query = args.query
    
    describe_image_command(image_path, query)
        

if __name__ == "__main__":
    main()