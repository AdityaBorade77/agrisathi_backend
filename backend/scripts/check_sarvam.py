import os, time, requests
from dotenv import load_dotenv

load_dotenv("backend/.env")
key = os.getenv("SARVAM_API_KEY")
print("Key found:", bool(key), "| length:", len(key) if key else 0)  # never prints the key

t = time.time()
r = requests.post(
    "https://api.sarvam.ai/v1/chat/completions",
    headers={"api-subscription-key": key},
    json={
        "model": "sarvam-105b",
        "messages": [{"role": "user", "content": "कांदा लागवडीसाठी कोणती माती चांगली? एका वाक्यात मराठीत उत्तर द्या."}],
        "max_tokens": 500,
        "reasoning_effort": None,
    },
    timeout=30,
)
print("Status:", r.status_code, "| time:", round(time.time() - t, 1), "s")
if r.ok:
    d = r.json()
    c = d["choices"][0]
    msg = c["message"]
    print("finish_reason:", c.get("finish_reason"))
    print("message fields:", list(msg.keys()))
    print("Answer:", (msg.get("content") or "<EMPTY>")[:300])
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    print("Reasoning length:", len(reasoning) if reasoning else 0)
    print("Tokens:", d.get("usage"))
else:
    print("Error:", r.text[:300])