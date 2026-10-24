# One-off script: extract the crop-relevant text of the ICAR Nashik
# contingency plan PDF into small chunks for BM25 retrieval.
# Run once:  python scripts/build_chunks.py
# Output (crop_api/data/chunks.json) is committed, so this script does not
# need to run again unless the source PDF or the page range changes.
import json
import os
import re

import pdfplumber

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FILE = os.path.join(BASE_DIR, "data", "sources", "ICAR_Nashik_2011.pdf")
OUT_FILE = os.path.join(BASE_DIR, "data", "chunks.json")

# Pages 1-13 (1-indexed) hold the district profile (sec 1.x) and the crop /
# vegetable / horticulture contingency tables about sowing, irrigation and
# soil (sec 2.0-2.2). From partway through page 14 onwards the plan moves
# into "Outbreak of pests and diseases due to unseasonal rains" -- pest and
# disease tables with chemical spray names and doses -- through page 17,
# then page 18 starts 2.3 Floods, 2.4 Extreme events and 2.5
# Livestock/Poultry/Fisheries. None of that belongs in crop recommendation
# evidence, so the plan stops at page 13. The SAFETY_FILTER below is a
# second line of defence in case any pesticide content slipped in anyway
# (e.g. a stray chemical mention inside pages 1-13).
FIRST_PAGE = 1
LAST_PAGE = 13

HEADING_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s")
MIN_CHUNK_CHARS = 600
MAX_CHUNK_CHARS = 900

# A crop recommendation must never surface a pesticide name or dose to a
# farmer unreviewed -- a wrong one could cause real harm. Any chunk that
# looks like a spray instruction gets dropped, not just truncated.
DOSE_UNITS = ["ml/L", "g/L", "kg/ha"]
CHEMICAL_WORDS = [
    "spraying", "endosulfan", "monocrotophos", "phorate", "chloropyriphos",
    "dimethoate", "mancozeb", "quinolphos", "parathion",
]


def is_pesticide_content(text):
    lower = text.lower()
    if any(word in lower for word in CHEMICAL_WORDS):
        return True
    return any(unit in text for unit in DOSE_UNITS)


def heading_at_or_before(headings, offset):
    """headings: list of (offset, section) sorted by offset. Returns the
    section whose heading offset is <= the given offset, or None."""
    section = None
    for h_offset, h_section in headings:
        if h_offset <= offset:
            section = h_section
        else:
            break
    return section


def split_page(text):
    """Yield (start, end) char spans of roughly MIN-MAX_CHUNK_CHARS each,
    preferring to cut on a line break so a chunk does not start or end
    mid-word."""
    start = 0
    length = len(text)
    while start < length:
        end = min(start + MAX_CHUNK_CHARS, length)
        if end < length:
            newline = text.rfind("\n", start + MIN_CHUNK_CHARS, end)
            if newline != -1:
                end = newline + 1
        yield start, end
        start = end


def chunks_for_page(page_num, text, carried_section):
    """Returns (list_of_chunks, new_carried_section) for one page."""
    headings = []
    offset = 0
    for line in text.split("\n"):
        match = HEADING_RE.match(line)
        if match:
            headings.append((offset, match.group(1)))
        offset += len(line) + 1  # +1 for the removed "\n"

    page_chunks = []
    for start, end in split_page(text):
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        section = heading_at_or_before(headings, start) or carried_section
        page_chunks.append({"page": page_num, "section": section, "text": chunk_text})

    last_heading = heading_at_or_before(headings, len(text))
    new_carried_section = last_heading or carried_section
    return page_chunks, new_carried_section


def main():
    all_chunks = []
    carried_section = None

    with pdfplumber.open(PDF_FILE) as pdf:
        for page_num in range(FIRST_PAGE, LAST_PAGE + 1):
            text = pdf.pages[page_num - 1].extract_text() or ""
            page_chunks, carried_section = chunks_for_page(page_num, text, carried_section)
            all_chunks.extend(page_chunks)

    before_filter = len(all_chunks)
    all_chunks = [c for c in all_chunks if not is_pesticide_content(c["text"])]
    dropped = before_filter - len(all_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = i

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Made {len(all_chunks)} chunks from pages {FIRST_PAGE}-{LAST_PAGE}. "
          f"Dropped {dropped} chunk(s) as pesticide/dose content. Saved to {OUT_FILE}\n")
    print("Sample chunks:")
    for chunk in (all_chunks[0], all_chunks[len(all_chunks) // 2], all_chunks[-1]):
        print(f"\n--- id={chunk['id']} page={chunk['page']} section={chunk['section']} "
              f"({len(chunk['text'])} chars) ---")
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()
