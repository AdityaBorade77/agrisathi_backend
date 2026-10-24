"""
check_contrast.py — check the app's colour pairs against WCAG 2.1 AA.

Reads the --c-* colour tokens from frontend/style.css (so it always
checks the real colours), then checks every foreground/background pair
the app uses.

  text pairs: need 4.5:1 (normal text, WCAG 1.4.3)
  ui pairs:   need 3.0:1 (input borders, focus ring, icons, WCAG 1.4.11)

Run (no extra packages needed, plain Python 3):
    python frontend/tools/check_contrast.py
Exit code 1 if any pair fails.
"""
import os
import re
import sys

CSS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style.css")

# (foreground token, background token, kind, where it is used)
PAIRS = [
    ("text",         "bg",          "text", "main text on page"),
    ("text",         "surface",     "text", "main text on cards"),
    ("text-muted",   "bg",          "text", "secondary text on page"),
    ("text-muted",   "surface",     "text", "secondary text on cards"),
    ("text-muted",   "green-light", "text", "audio time inside bot bubble"),
    ("on-dark",      "green",       "text", "primary button"),
    ("on-dark",      "green-dark",  "text", "top bar"),
    ("on-dark",      "soil",        "text", "crop button"),
    ("on-dark",      "soil-dark",   "text", "crop button (pressed)"),
    ("on-dark",      "ochre",       "text", "disease button"),
    ("on-dark",      "red",         "text", "recording mic / error button"),
    ("green-dark",   "green-light", "text", "bot bubble, active nav item"),
    ("green-dark",   "surface",     "text", "secondary button, result crop name"),
    ("soil-dark",    "soil-light",  "text", "farmer bubble"),
    ("soil-dark",    "surface",     "text", "crop months"),
    ("text",         "ochre-light", "text", "offline banner, risk box"),
    ("ochre",        "ochre-light", "text", "risk label"),
    ("text",         "red-light",   "text", "error box"),
    ("red",          "red-light",   "text", "listening text"),
    ("green-dark",   "bg",          "text", "links"),
    ("border-strong","surface",     "ui",   "input / choice button border"),
    ("border-strong","bg",          "ui",   "language button border"),
    ("green",        "surface",     "ui",   "selected input border, progress bar"),
    ("focus",        "bg",          "ui",   "keyboard focus ring (page)"),
    ("focus",        "surface",     "ui",   "keyboard focus ring (cards)"),
    ("focus-on-dark","green-dark",  "ui",   "keyboard focus ring (top bar)"),
    ("red",          "surface",     "ui",   "error box border"),
]

NEEDED = {"text": 4.5, "ui": 3.0}


def read_tokens(css_text: str) -> dict:
    """Return {'bg': '#FAF6EE', ...} for every --c-name: #RRGGBB; line."""
    return {name: value.upper() for name, value in re.findall(r"--c-([a-z-]+):\s*(#[0-9A-Fa-f]{6})\b", css_text)}


def luminance(hex_colour: str) -> float:
    """Relative luminance, WCAG 2.1 formula."""
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def main() -> int:
    with open(CSS_PATH, encoding="utf-8") as f:
        tokens = read_tokens(f.read())

    failures = 0
    print(f"{'foreground':<14} {'background':<12} {'ratio':>6}  {'need':>4}  result  used for")
    print("-" * 86)
    for fg, bg, kind, used_for in PAIRS:
        if fg not in tokens or bg not in tokens:
            print(f"{fg:<14} {bg:<12}  token missing in style.css")
            failures += 1
            continue
        ratio = contrast(tokens[fg], tokens[bg])
        ok = ratio >= NEEDED[kind]
        failures += 0 if ok else 1
        print(f"{fg:<14} {bg:<12} {ratio:6.2f}  {NEEDED[kind]:4.1f}  {'PASS' if ok else 'FAIL':<6}  {used_for}")

    print("-" * 86)
    print("All pairs pass WCAG AA." if failures == 0 else f"{failures} pair(s) FAIL.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
