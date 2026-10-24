# Sends one leaf photo to the running API.
# Usage:  python test_api.py "C:\path\to\leaf.jpg" Onion
import json
import sys

import requests

URL = "http://127.0.0.1:5002/predict-disease"

if len(sys.argv) < 3:
    print('Usage: python test_api.py "path\\to\\leaf.jpg" CropName')
    sys.exit(1)

photo, crop = sys.argv[1], sys.argv[2]
with open(photo, "rb") as f:
    reply = requests.post(URL, files={"image": f}, data={"crop": crop}, timeout=60)

print("HTTP status:", reply.status_code)
print(json.dumps(reply.json(), indent=2))
