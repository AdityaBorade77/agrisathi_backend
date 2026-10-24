"""
utils/text_cleaner.py — LLM Response Text Cleaner
===================================================
Cleans LLM-generated text before it is sent to the translation API.

Pipeline:
    LLM Response → clean_for_translation() → Translation API

What is removed:
    - Emojis
    - Markdown bold/italic/underline formatting
    - Markdown headings (#, ##, ...)
    - Markdown bullet markers at line starts (-, *, +)
    - Inline code backticks
    - Markdown hyperlinks  [text](url) → text
    - Decorative Unicode symbols (bullets, arrows, etc.)
    - Excessive whitespace / blank lines

What is PRESERVED:
    - All legitimate Unicode text: English, Hindi, Marathi, Gujarati,
      Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi, Urdu, etc.
    - Useful punctuation: . , ! ? ; : ( ) / % ' " -
    - En-dash (–) and em-dash (—) inside sentences
    - Meaningful paragraph breaks (up to one blank line)
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decorative / non-text Unicode symbols to strip.
# We keep en-dash (U+2013) and em-dash (U+2014) because they appear inside
# sentences (e.g. "Powdery mildew – White, dusty patches").
# ---------------------------------------------------------------------------
_DECORATIVE_SYMBOLS_RE = re.compile(
    "[\u2022"   # • BULLET
    "\u25A0"    # ■ BLACK SQUARE
    "\u25CF"    # ● BLACK CIRCLE
    "\u25B6"    # ▶ BLACK RIGHT-POINTING TRIANGLE
    "\u25C0"    # ◀ BLACK LEFT-POINTING TRIANGLE
    "\u2192"    # → RIGHT ARROW
    "\u2190"    # ← LEFT ARROW
    "\u21B3"    # ↳ DOWNWARDS ARROW WITH TIP RIGHTWARDS
    "\u2764"    # ❤ HEAVY BLACK HEART (non-emoji version)
    "\u2665"    # ♥ BLACK HEART SUIT
    "\u2605"    # ★ BLACK STAR
    "\u2606"    # ☆ WHITE STAR
    "\u2713"    # ✓ CHECK MARK
    "\u2714"    # ✔ HEAVY CHECK MARK
    "\u2717"    # ✗ BALLOT X
    "\u2718"    # ✘ HEAVY BALLOT X
    "\u00BB"    # » RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK (decorative)
    "\u00AB"    # « LEFT-POINTING DOUBLE ANGLE QUOTATION MARK (decorative)
    "]"
)


def clean_for_translation(text: str) -> str:
    """
    Clean LLM-generated text so it is safe to send to the translation API.

    Args:
        text: Raw LLM response string (may contain markdown, emojis, etc.)

    Returns:
        Cleaned plain-text string with language characters fully preserved.
        If text is empty/None, returns "".
        If an unexpected error occurs, logs it and returns the original text
        so the translation pipeline is never left with nothing.

    Examples:
        >>> clean_for_translation("**Hello** 🌱 _world_")
        'Hello world'
        >>> clean_for_translation("पत्तियों पर सफेद धब्बे दिखाई दे रहे हैं।")
        'पत्तियों पर सफेद धब्बे दिखाई दे रहे हैं।'
    """
    if not text:
        return ""

    original = text  # kept for fallback

    try:
        # ── 1. Remove emojis ──────────────────────────────────────────────
        # The `emoji` library correctly targets only emoji code points,
        # leaving all Indic/Latin/other Unicode script characters intact.
        try:
            import emoji as _emoji_lib
            text = _emoji_lib.replace_emoji(text, replace="")
        except ImportError:
            # Fallback: strip by Unicode category if `emoji` is unavailable.
            text = "".join(
                ch for ch in text
                if unicodedata.category(ch) not in ("So",)
                or unicodedata.name(ch, "").startswith("LATIN")
            )
            logger.warning("'emoji' package not installed; using fallback emoji stripper.")

        # ── 2. Remove inline code backticks:  `text` → text ──────────────
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # ── 3. Remove Markdown links:  [label](url) → label ──────────────
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

        # ── 4. Bold: **text** → text  and  __text__ → text ───────────────
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)

        # ── 5. Italic: *text* → text  (single asterisk, not double) ──────
        text = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", text)

        # ── 6. Italic underscore: _text_ → text ───────────────────────────
        # Only match at word boundaries to avoid breaking compound words.
        text = re.sub(r"(?<!\w)_(?!\s)(.*?)(?<!\s)_(?!\w)", r"\1", text)

        # ── 7. Markdown headings:  ## Heading → Heading ───────────────────
        text = re.sub(r"^#{1,6}\s+(.*)", r"\1", text, flags=re.MULTILINE)

        # ── 8. Bullet markers at start of lines:  - Item / * Item / + Item ─
        # Requires at least one space after marker so lone hyphens inside
        # sentences (e.g. "well-known") are never stripped.
        text = re.sub(r"^[ \t]*[-*+][ \t]+", "", text, flags=re.MULTILINE)

        # ── 9. Decorative Unicode symbols ────────────────────────────────
        text = _DECORATIVE_SYMBOLS_RE.sub("", text)

        # ── 10. Whitespace normalisation ──────────────────────────────────
        # a) Strip leading whitespace from each line (emoji removal can leave
        #    a stray leading space when an emoji was at the start of a line)
        text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)

        # b) Collapse multiple spaces / tabs on a single line → one space
        text = re.sub(r"[ \t]+", " ", text)

        # c) Strip trailing spaces from each line
        text = re.sub(r" +$", "", text, flags=re.MULTILINE)

        # d) Collapse 3+ consecutive blank lines → at most one blank line
        text = re.sub(r"\n{3,}", "\n\n", text)

        # e) Strip leading/trailing whitespace from the whole string
        text = text.strip()

    except Exception as exc:
        # Safety net: log and return original so the pipeline still has content.
        logger.error("clean_for_translation() failed unexpectedly: %s", exc, exc_info=True)
        return original

    return text


# ---------------------------------------------------------------------------
# Self-contained test suite
# Run with:  python backend/utils/text_cleaner.py
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Lightweight test suite — prints PASS / FAIL for each case."""
    import sys

    cases = [
        # (description, input, expected_output)

        # ── English + emojis ─────────────────────────────────────────────
        (
            "English with emojis",
            "Check this out 🔍 it is very useful 💡",
            # Emojis removed → adjacent spaces collapse to one space
            "Check this out it is very useful",
        ),

        # ── Markdown bold ────────────────────────────────────────────────
        (
            "Markdown bold",
            "**Hello** world and **Bold phrase**",
            "Hello world and Bold phrase",
        ),

        # ── Markdown italic ──────────────────────────────────────────────
        (
            "Markdown italic (asterisk)",
            "*Italic* text here",
            "Italic text here",
        ),

        # ── Markdown heading ─────────────────────────────────────────────
        (
            "Markdown headings",
            "## Section Title\n### Sub-section",
            "Section Title\nSub-section",
        ),

        # ── Bullet points ────────────────────────────────────────────────
        (
            "Bullet markers removed",
            "- Apple\n* Banana\n+ Cherry",
            "Apple\nBanana\nCherry",
        ),

        # ── Normal hyphens preserved ─────────────────────────────────────
        (
            "Normal hyphen inside sentence preserved",
            "well-known fact and state-of-the-art model",
            "well-known fact and state-of-the-art model",
        ),

        # ── En-dash inside sentence preserved ────────────────────────────
        (
            "En-dash inside sentence preserved",
            "Powdery mildew \u2013 White, dusty patches that can spread.",
            "Powdery mildew \u2013 White, dusty patches that can spread.",
        ),

        # ── Inline code ──────────────────────────────────────────────────
        (
            "Inline code backticks",
            "Use `pip install` to install",
            "Use pip install to install",
        ),

        # ── Markdown link ────────────────────────────────────────────────
        (
            "Markdown link",
            "Visit [Google](https://google.com) for more info",
            "Visit Google for more info",
        ),

        # ── Hindi (must be preserved exactly) ────────────────────────────
        (
            "Hindi text preserved",
            "\u092a\u0924\u094d\u0924\u093f\u092f\u094b\u0902 \u092a\u0930 \u0938\u092b\u0947\u0926 \u0927\u092c\u094d\u092c\u0947 \u0926\u093f\u0916\u093e\u0908 \u0926\u0947 \u0930\u0939\u0947 \u0939\u0948\u0902\u0964",
            "\u092a\u0924\u094d\u0924\u093f\u092f\u094b\u0902 \u092a\u0930 \u0938\u092b\u0947\u0926 \u0927\u092c\u094d\u092c\u0947 \u0926\u093f\u0916\u093e\u0908 \u0926\u0947 \u0930\u0939\u0947 \u0939\u0948\u0902\u0964",
        ),

        # ── Marathi (must be preserved exactly) ──────────────────────────
        (
            "Marathi text preserved",
            "\u092a\u093e\u0928\u093e\u0902\u0935\u0930 \u092a\u093e\u0902\u0922\u0930\u0947 \u0921\u093e\u0917 \u0926\u093f\u0938\u0924 \u0906\u0939\u0947\u0924.",
            "\u092a\u093e\u0928\u093e\u0902\u0935\u0930 \u092a\u093e\u0902\u0922\u0930\u0947 \u0921\u093e\u0917 \u0926\u093f\u0938\u0924 \u0906\u0939\u0947\u0924.",
        ),

        # ── Mixed English + Hindi ─────────────────────────────────────────
        (
            "Mixed English + Hindi",
            "**Diagnosis:** \u092a\u0924\u094d\u0924\u093f\u092f\u094b\u0902 \u092a\u0930 \u0938\u092b\u0947\u0926 \u0927\u092c\u094d\u092c\u0947 \U0001f331",
            "Diagnosis: \u092a\u0924\u094d\u0924\u093f\u092f\u094b\u0902 \u092a\u0930 \u0938\u092b\u0947\u0926 \u0927\u092c\u094d\u092c\u0947",
        ),

        # ── Mixed English + Marathi ───────────────────────────────────────
        (
            "Mixed English + Marathi",
            "## Result\n- \u092a\u093e\u0928\u093e\u0902\u0935\u0930 \u092a\u093e\u0902\u0922\u0930\u0947 \u0921\u093e\u0917 \u0926\u093f\u0938\u0924 \u0906\u0939\u0947\u0924.",
            "Result\n\u092a\u093e\u0928\u093e\u0902\u0935\u0930 \u092a\u093e\u0902\u0922\u0930\u0947 \u0921\u093e\u0917 \u0926\u093f\u0938\u0924 \u0906\u0939\u0947\u0924.",
        ),

        # ── Excessive whitespace ──────────────────────────────────────────
        (
            "Excessive whitespace normalised",
            "Hello     world\n\n\n\nNew paragraph",
            "Hello world\n\nNew paragraph",
        ),

        # ── Full LLM example ─────────────────────────────────────────────
        (
            "Full LLM markdown example",
            (
                "White spots on leaves can come from several different causes. "
                "Here are the most common ones: \U0001f50d\n\n"
                "**Likely causes:**\n\n"
                "* **Powdery mildew** \u2013 White, dusty or floury patches that can spread.\n"
                "* **Spider mites** \u2013 Tiny white/yellow speckles.\n"
                "* **Mealybugs or scale** \u2013 Small white cottony dots.\n\n"
                "\U0001f6e0 **To help identify it:**\n\n"
                "* Can you wipe the spots off easily?\n"
                "* Any webbing, stickiness, or yellowing elsewhere?\n\n"
                "\U0001f4a1 **Quick steps:**\n\n"
                "- Isolate the plant.\n"
                "+ Improve airflow.\n"
                "- Avoid overhead watering."
            ),
            (
                "White spots on leaves can come from several different causes. "
                "Here are the most common ones:\n\n"
                "Likely causes:\n\n"
                "Powdery mildew \u2013 White, dusty or floury patches that can spread.\n"
                "Spider mites \u2013 Tiny white/yellow speckles.\n"
                "Mealybugs or scale \u2013 Small white cottony dots.\n\n"
                "To help identify it:\n\n"
                "Can you wipe the spots off easily?\n"
                "Any webbing, stickiness, or yellowing elsewhere?\n\n"
                "Quick steps:\n\n"
                "Isolate the plant.\n"
                "Improve airflow.\n"
                "Avoid overhead watering."
            ),
        ),

        # ── Empty / null safety ───────────────────────────────────────────
        (
            "Empty string returns empty",
            "",
            "",
        ),
        (
            "None-like falsy returns empty",
            None,
            "",
        ),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("  text_cleaner.py — Test Suite")
    print("=" * 60)

    for desc, inp, expected in cases:
        result = clean_for_translation(inp)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {desc}")
        if not ok:
            failed += 1
            print(f"  INPUT    : {repr(inp)}")
            print(f"  EXPECTED : {repr(expected)}")
            print(f"  GOT      : {repr(result)}")
        else:
            passed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_tests()
