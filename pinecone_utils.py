import os
import logging
from pinecone import Pinecone

logger = logging.getLogger("pinecone_utils")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")

if not PINECONE_API_KEY or not PINECONE_INDEX:
    raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set in the environment variables.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def _sanitize_metadata(meta: dict) -> dict:
    safe = {}
    for k, v in meta.items():
        if v is None:
            safe[k] = ""
        elif isinstance(v, (str, bool, float, int)):
            safe[k] = v
        elif isinstance(v, list):
            safe[k] = [str(x) for x in v]
        else:
            safe[k] = str(v)
    return safe

def upsert_listings(listings):
    vectors = []
    for listing in listings:
        if "embedding" not in listing or not listing["embedding"]:
            logger.warning(f"Listing {listing.get('id')} has no embedding, skipping.")
            continue
        meta = dict(listing)
        emb = meta.pop("embedding")
        meta = _sanitize_metadata(meta)
        vectors.append({
            "id": str(meta.get("id")),
            "values": emb,
            "metadata": meta
        })
    logger.info(f"Prepared {len(vectors)} vectors for upsert")
    if not vectors:
        logger.warning("No data to upsert into Pinecone!")
        return
    index.upsert(vectors=vectors)
    logger.info(f"Upserted {len(vectors)} vectors to Pinecone")

def semantic_search(query_embedding, top_k=5):
    logger.info("semantic_search called")
    if not query_embedding or not isinstance(query_embedding, list) or not len(query_embedding):
        logger.error(f"Invalid query embedding: {query_embedding}")
        return []
    try:
        res = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        results = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        logger.info(f"Semantic search returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Pinecone semantic search error: {e}")
        return []
