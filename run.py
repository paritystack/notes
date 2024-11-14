#! /usr/bin/env python3
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="For creating missing files in the src directory")
    parser.add_argument("--file", default="src/SUMMARY.md", type=str, help="Input file")
    args = parser.parse_args()
    #print(f"Input file: {args.file}")

    with open(args.file, "r") as f:
        content = f.read()
        
        for line in content.split("\n"):
            if "(" in line and ")" in line:
                filename = line[line.find("(")+1:line.find(")")]
                #print(filename)
                if filename.endswith(".md"):
                    #print(f"Processing {filename}")
                    try:
                        with open(f"src/{filename}", "r") as f:
                            pass
                    except FileNotFoundError:
                        # Create directory if it doesn't exist
                        directory = f"src/{'/'.join(filename.split('/')[:-1])}"
                        if directory != "src":  # Only create dir if there's a subdirectory
                            os.makedirs(directory, exist_ok=True)
                            
                        # Create the file
                        with open(f"src/{filename}", "w") as f:
                            pass
                        print(f"Created empty file: src/{filename}")


if __name__ == "__main__":
    main()
