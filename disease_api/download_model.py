# Download the trained model files into disease_api/model/ (used on Render,
# where model/ is not in git because best_model.pt is large).
#
# Set MODEL_BASE_URL to a folder that holds best_model.pt and model_meta.json,
# e.g. a Hugging Face model repo:
#   https://huggingface.co/<your-user>/<repo-name>/resolve/main
# If that repo is private, also set HF_TOKEN (read access).
#
# Run:  python download_model.py
# Files that already exist are left alone, so on your laptop this does nothing.
import os
import sys
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FILES = ["best_model.pt", "model_meta.json"]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    missing = [f for f in FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if not missing:
        print("Model files already present, nothing to download.")
        return

    # strip(): a value pasted into a dashboard often carries a hidden
    # space or line break at the end, which breaks the URL / token
    base_url = os.getenv("MODEL_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        sys.exit(f"Missing {missing} and MODEL_BASE_URL is not set.")

    token = (os.getenv("HF_TOKEN") or "").strip()
    for name in missing:
        request = urllib.request.Request(f"{base_url}/{name}")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        target = os.path.join(MODEL_DIR, name)
        print(f"Downloading {name} ...")
        try:
            with urllib.request.urlopen(request, timeout=300) as response, open(target + ".part", "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        except urllib.error.HTTPError as e:
            # Explain the usual causes. Never print the token itself.
            print(f"\nFAILED: HTTP {e.code} for {base_url}/{name}")
            print(f"HF_TOKEN sent: {'yes' if token else 'NO (not set on this service)'}")
            if e.code in (401, 403, 404):
                print("Check: 1) the URL uses /resolve/main and the user/repo names are exact;")
                print("       2) if the repo is private, HF_TOKEN is set on THIS Render service")
                print("          and belongs to an account that can read the repo.")
            sys.exit(1)
        # Rename only after a complete download, so a broken download never looks finished
        os.replace(target + ".part", target)
        print(f"  saved {os.path.getsize(target) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
