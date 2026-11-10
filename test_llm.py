import os, httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY", "").strip()
base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
model = os.getenv("OPENAI_MODEL", "")
print("key length:", len(key), "| starts with:", key[:4], "| model:", model, "| base:", base)

r = httpx.post(
    f"{base}/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    timeout=30,
)
print(r.status_code)
print(r.text[:400])