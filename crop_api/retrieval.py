# BM25 keyword retrieval over chunks of the ICAR Nashik contingency plan.
# No vector database, no embeddings -- just lexical matching (rank_bm25),
# which is enough for short, English, table-heavy agronomy text and is
# easy to explain in the report.
import json
import os
import re

from rank_bm25 import BM25Okapi

from rules import load_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks.json")

TOKEN_RE = re.compile(r"\w+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_chunks(path=CHUNKS_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def crop_terms_from_rules(rules):
    return {crop["id"]: crop.get("search_terms") or [] for crop in rules}


def build_index(chunks):
    if not chunks:
        return None
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


# Loaded once at startup (app.py imports this module when it starts).
_chunks = load_chunks()
_crop_terms = crop_terms_from_rules(load_rules())
_bm25 = build_index(_chunks)


def retrieve(crop_ids, extra_terms=None, k=3, chunks=None, crop_terms=None):
    """Returns up to k chunks as {page, section, text, score}, ranked by
    BM25 over the crops' search_terms plus extra_terms (e.g. season, soil
    words). Only chunks containing at least one of the crops' search_terms
    are returned -- extra_terms only influence ranking, not inclusion.

    chunks/crop_terms let tests inject a small fake dataset instead of the
    real chunks.json / crop_rules.json.
    """
    if isinstance(crop_ids, str):
        crop_ids = [crop_ids]

    use_chunks = _chunks if chunks is None else chunks
    use_terms = _crop_terms if crop_terms is None else crop_terms
    bm25 = _bm25 if chunks is None else build_index(use_chunks)

    search_terms = []
    for crop_id in crop_ids:
        search_terms.extend(use_terms.get(crop_id, []))
    if not search_terms or bm25 is None:
        return []

    query_terms = search_terms + list(extra_terms or [])
    scores = bm25.get_scores(tokenize(" ".join(query_terms)))

    lower_terms = [term.lower() for term in search_terms]
    candidates = [
        (score, chunk) for score, chunk in zip(scores, use_chunks)
        if any(term in chunk["text"].lower() for term in lower_terms)
    ]
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {"page": chunk["page"], "section": chunk["section"], "text": chunk["text"], "score": round(float(score), 4)}
        for score, chunk in candidates[:k]
    ]
