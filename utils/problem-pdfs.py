"""Render one PDF per problem (statement and solution) from the soft-matter problems file.

Usage: python utils/problem-pdfs.py [problems.qmd] [output_dir]
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "quarto/soft-matter/soft-matter_problems.qmd"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "quarto/soft-matter/problem-pdfs"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def problems(text):
    # Headings inside ::: blocks (the callouts' "## Solution") are not section breaks.
    chapter, title, body, depth = None, None, [], 0
    for line in text.splitlines():
        fence = re.match(r"\s*(:::+)\s*(\{)?", line)
        if fence:
            depth += 1 if fence.group(2) else -1
        elif depth == 0 and re.match(r"#{2,3} ", line):
            if title:
                yield chapter, title, "\n".join(body).strip()
            title, body = None, []
            if line.startswith("### "):
                title = line[4:].strip()
            else:
                chapter = line[3:].strip()
            continue
        body.append(line)
    if title:
        yield chapter, title, "\n".join(body).strip()


out.mkdir(parents=True, exist_ok=True)
# Render outside the project, so the book's _quarto.yml does not apply.
with tempfile.TemporaryDirectory() as tmp:
    for i, (chapter, title, body) in enumerate(problems(src.read_text(encoding="utf-8")), start=1):
        body = re.sub(r'(:::+\s*\{\s*\.callout-note)\s+collapse="true"\s*\}', r"\1}", body)
        name = f"{i:02d}-{slug(chapter)}-{slug(title)}"
        qmd = Path(tmp) / f"{name}.qmd"
        qmd.write_text(
            f'---\ntitle: "{title}"\nsubtitle: "Complex disordered systems: {chapter}"\n'
            "format:\n  pdf:\n    documentclass: article\n    geometry: margin=2.5cm\n---\n\n" + body + "\n",
            encoding="utf-8",
        )
        subprocess.run(["quarto", "render", qmd.name, "--to", "pdf", "--quiet"], cwd=tmp, check=True)
        shutil.copy(qmd.with_suffix(".pdf"), out / f"{name}.pdf")
        print(out / f"{name}.pdf")
