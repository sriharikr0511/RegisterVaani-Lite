import os, httpx
from dotenv import load_dotenv

load_dotenv()
r = httpx.get(
    os.getenv("OPENAI_BASE_URL").rstrip("/") + "/models",
    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY').strip()}"},
    timeout=30,
)
print(r.status_code)
for m in sorted(x["id"] for x in r.json().get("data", [])):
    print(m)