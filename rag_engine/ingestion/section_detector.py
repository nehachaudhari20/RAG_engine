# rag_engine/ingestion/section_detector.py
import re
from typing import List, Tuple

SECTION_REGEX = re.compile(
    r"^(\d+(\.\d+)*)\s+[A-Z][A-Za-z0-9\s\-]{2,}$"
)


def detect_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.split("\n")
    sections = []

    current_title = "Abstract"
    current_lines = []

    for line in lines:
        line = line.strip()

        if SECTION_REGEX.match(line):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))

    return sections
