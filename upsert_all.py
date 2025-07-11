import json
from llm_utils import get_embedding, generate_summary, llm_parse_listing
from pinecone_utils import upsert_listings

with open("parser/my_listings.json", "r", encoding="utf-8") as f:
    all_listings = json.load(f)

enriched = []
for data in all_listings:
    full_desc = f"{data.get('description','')}\nPrice: {data.get('price','')}\nFurnished: {data.get('furnished','')}"
    if "pets_allowed" in data:
        full_desc += f"\nPets allowed: {data['pets_allowed']}"
    try:
        llm_fields_raw = llm_parse_listing(full_desc, lang="English")
        llm_fields = json.loads(llm_fields_raw)
    except Exception:
        llm_fields = {}
    summary = generate_summary(full_desc, lang="English")
    try:
        embedding = get_embedding(full_desc)
    except Exception:
        embedding = []
    listing = {
        "id": data.get("url"),
        "url": data.get("url"),
        "title": data.get("title") or llm_fields.get("title") or "",
        "price": str(data.get("price") or llm_fields.get("price") or ""),
        "district": data.get("district") or llm_fields.get("district") or "",
        "description": data.get("description") or "",
        "furnished": data.get("furnished") if "furnished" in data else llm_fields.get("furnished"),
        "pets_allowed": data.get("pets_allowed") if "pets_allowed" in data else llm_fields.get("pets_allowed"),
        "summary": summary,
        "embedding": embedding,
    }
    enriched.append(listing)

print(f"Uploading {len(enriched)} listings to Pinecone...")
upsert_listings(enriched)
print("Done.")
