import os, json, requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
INDEX_NAME = (os.getenv("PINECONE_INDEX") or "").strip()
HOST = (os.getenv("PINECONE_HOST") or "").strip()

if not API_KEY or not INDEX_NAME or not HOST:
    raise ValueError("PINECONE_API_KEY, PINECONE_INDEX and PINECONE_HOST must be set.")

def _clean_host(h: str) -> str:
    h = h.strip().replace(" ", "")
    if "://" in h:
        h = h.split("://", 1)[1]
    h = h.split("/", 1)[0]
    if "-UNKNOWN" in h:
        h = h.split("-UNKNOWN", 1)[0]
    return h

BASE = "https://" + _clean_host(HOST)
HEADERS = {"Api-Key": API_KEY, "Content-Type": "application/json"}

def _san(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, bool, float, int)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = str(v)
    return out

def upsert_listings(listings):
    vectors = []
    for row in listings or []:
        emb = row.get("embedding")
        if not emb:
            continue
        meta = dict(row)
        meta.pop("embedding", None)
        meta = _san(meta)
        vid = str(meta.get("id") or meta.get("url") or len(vectors) + 1)
        vectors.append({"id": vid, "values": emb, "metadata": meta})
    if not vectors:
        return {"upserted": 0}
    r = requests.post(f"{BASE}/vectors/upsert", headers=HEADERS, data=json.dumps({"vectors": vectors}), timeout=30)
    r.raise_for_status()
    data = r.json()
    return {"upserted": data.get("upserted_count") or data.get("upserted") or 0}

def semantic_search(query_embedding, top_k=5, filters=None):
    if not isinstance(query_embedding, list) or not query_embedding:
        return []
    payload = {
        "vector": query_embedding,
        "topK": int(top_k),
        "includeValues": False,
        "includeMetadata": True
    }
    if filters:
        payload["filter"] = filters
    r = requests.post(f"{BASE}/query", headers=HEADERS, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("matches", []) or []
