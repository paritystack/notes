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

def add_to_summary(summary_file, files_to_add):
    """Add missing file references to SUMMARY.md"""
    if not files_to_add:
        return

    with open(summary_file, "r") as f:
        lines = f.readlines()

    # Group files by directory
    files_by_dir = {}
    for file in files_to_add:
        dir_name = file.split('/')[0] if '/' in file else 'misc'
        if dir_name not in files_by_dir:
            files_by_dir[dir_name] = []
        files_by_dir[dir_name].append(file)

    # Find sections and add files
    modified = False
    for dir_name, files in files_by_dir.items():
        # Find the section for this directory
        section_start = None
        section_end = None

        for i, line in enumerate(lines):
            # Look for section header (e.g., "- [Programming Languages](programming/README.md)")
            if f"({dir_name}/README.md)" in line or f"[{dir_name.replace('_', ' ').title()}]" in line:
                section_start = i
                # Find the end of this section (next section or end of file)
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('- [') or lines[j].startswith('* ['):
                        section_end = j
                        break
                if section_end is None:
                    section_end = len(lines)
                break

        if section_start is not None:
            # Add each file to the section
            for file in sorted(files):
                # Extract filename without extension for title
                filename = file.split('/')[-1].replace('.md', '').replace('_', ' ').title()
                new_line = f"    - [{filename}]({file})\n"

                # Insert before the section end
                lines.insert(section_end, new_line)
                section_end += 1
                modified = True
                print(f"Added to SUMMARY.md: {file}")
        else:
            print(f"Warning: Could not find section for {dir_name}, skipping files: {files}")

    if modified:
        with open(summary_file, "w") as f:
            f.writelines(lines)

def update_operation(summary_file):
    """Create missing files and add unreferenced files to SUMMARY.md"""
    summary_files = set(get_files_from_summary(summary_file))
    directory_files = set(get_files_from_directory("src"))

    # Part 1: Create missing files referenced in SUMMARY.md
    missing_in_directory = summary_files - directory_files
    if missing_in_directory:
        print("Creating missing files referenced in SUMMARY.md:")
        for filename in sorted(missing_in_directory):
            # Create directory if it doesn't exist
            directory = f"src/{'/'.join(filename.split('/')[:-1])}"
            if directory != "src":  # Only create dir if there's a subdirectory
                os.makedirs(directory, exist_ok=True)

            # Create the file
            with open(f"src/{filename}", "w") as f:
                pass
            print(f"  Created: src/{filename}")
        print()

    # Part 2: Add unreferenced files to SUMMARY.md
    missing_in_summary = directory_files - summary_files
    if missing_in_summary:
        print("Adding unreferenced files to SUMMARY.md:")
        add_to_summary(summary_file, sorted(missing_in_summary))

    if not missing_in_directory and not missing_in_summary:
        print("Everything is in sync - no updates needed.")

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
