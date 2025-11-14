#! /usr/bin/env python3
import argparse
import os

def get_files_from_summary(summary_file):
    """Extract all .md file references from SUMMARY.md"""
    files = []
    with open(summary_file, "r") as f:
        content = f.read()
        for line in content.split("\n"):
            if "(" in line and ")" in line:
                filename = line[line.find("(")+1:line.find(")")]
                if filename.endswith(".md"):
                    files.append(filename)
    return files

def get_files_from_directory(base_dir="src"):
    """Get all .md files in the src directory (excluding SUMMARY.md)"""
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".md") and filename != "SUMMARY.md":
                # Get relative path from src directory
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, base_dir)
                files.append(relative_path)
    return files

def check_operation(summary_file):
    """Check for missing files and unreferenced files"""
    summary_files = set(get_files_from_summary(summary_file))
    directory_files = set(get_files_from_directory("src"))

    missing_in_directory = summary_files - directory_files
    missing_in_summary = directory_files - summary_files

    if missing_in_directory:
        print("Files referenced in SUMMARY.md but missing in directory:")
        for file in sorted(missing_in_directory):
            print(f"  - {file}")
    else:
        print("All files referenced in SUMMARY.md exist in directory.")

    print()

    if missing_in_summary:
        print("Files in directory but not referenced in SUMMARY.md:")
        for file in sorted(missing_in_summary):
            print(f"  - {file}")
    else:
        print("All files in directory are referenced in SUMMARY.md.")

def update_operation(summary_file):
    """Create missing files referenced in SUMMARY.md"""
    files = get_files_from_summary(summary_file)

    for filename in files:
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

def main():
    parser = argparse.ArgumentParser(description="For creating missing files in the src directory")
    parser.add_argument("--file", default="src/SUMMARY.md", type=str, help="Input file")
    parser.add_argument("--op", default="update", type=str, choices=["check", "update"],
                        help="Operation: 'check' to compare files, 'update' to create missing files")
    args = parser.parse_args()

    if args.op == "check":
        check_operation(args.file)
    elif args.op == "update":
        update_operation(args.file)


if __name__ == "__main__":
    main()
