import argparse
import re

def split_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Use regex to split only at '## ' (not '###' or deeper)
    sections = re.split(r"(?=\n## )", content)

    if len(sections) == 1:
        print("No '##' headers found. Exiting.")
        return

    for i, section in enumerate(sections, start=1):
        output_file = f"{file_path.rsplit('.', 1)[0]}_part_{i}.qmd"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(section.strip() + "\n")
        print(f"Created: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a Markdown file at every '##' heading.")
    parser.add_argument("file", help="Path to the Markdown file to split")
    
    args = parser.parse_args()
    split_markdown(args.file)
