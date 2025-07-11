import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector for the input text (returns list of floats)."""
    if not text:
        return []
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding

def generate_summary(description, lang="English"):
    """Generate 1–2 sentence summary of the listing for search display."""
    if not description:
        return ""
    prompt = {
        "English": "Summarize this rental property for a search result (1-2 sentences, focus on main features and location):\n",
        "Italiano": "Riassumi questo annuncio immobiliare per un risultato di ricerca (1-2 frasi, indica i punti principali e la zona):\n"
    }.get(lang, "Summarize this rental property for a search result (1-2 sentences):\n")
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": description}
        ],
        temperature=0.3,
        max_tokens=70
    )
    return completion.choices[0].message.content.strip()

def llm_parse_listing(description, lang="English"):
    """Extract key fields (title, district, price, pets_allowed, furnished) as JSON from the listing description."""
    if not description:
        return "{}"
    prompt = {
        "English": """Extract these fields from the rental listing as JSON: 
- title
- district (neighborhood or area)
- price
- pets_allowed (true/false if possible)
- furnished (true/false if possible)
If information is missing, use null. Output JSON only.
Listing:\n""",
        "Italiano": """Estrai questi campi dall'annuncio immobiliare come JSON: 
- title
- district (zona o quartiere)
- price
- pets_allowed (true/false se possibile)
- furnished (true/false se possibile)
Se mancano informazioni, usa null. Solo output JSON.
Annuncio:\n"""
    }[lang]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": description}
        ],
        temperature=0,
        max_tokens=180,
    )
    return response.choices[0].message.content.strip()

def explain_match(query, metadata, lang="English", mode="fast"):
    """
    Explain why this listing matches the search query.
    mode="llm": Use OpenAI for smart, human-like explanations (recommended for demo/final version)
    mode="fast": Use simple keyword overlap
    """
    if mode == "llm":
        try:
            sys_prompt = {
                "English": "You are an assistant who explains in 1-2 sentences why this rental listing is a good match for the query below.",
                "Italiano": "Sei un assistente che spiega in 1-2 frasi perché questo annuncio immobiliare corrisponde alla richiesta dell'utente."
            }.get(lang, "You are an assistant who explains in 1-2 sentences why this rental listing is a good match for the query below.")
            user_prompt = {
                "English": f"Query: {query}\nListing: {metadata}",
                "Italiano": f"Richiesta: {query}\nAnnuncio: {metadata}"
            }.get(lang, f"Query: {query}\nListing: {metadata}")
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=80,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM explanation error: {e}"

    # Fast mode 
    try:
        meta_str = " ".join([
            str(metadata.get("title", "")),
            str(metadata.get("description", "")),
            str(metadata.get("summary", "")),
            str(metadata.get("district", "")),
            str(metadata.get("price", "")),
        ]).lower()
        query_str = query.lower()
        matches = [w for w in query_str.split() if w in meta_str]
        explanation = {
            "English": "Matched keywords: ",
            "Italiano": "Parole chiave trovate: "
        }.get(lang, "Matched keywords: ") + (", ".join(matches) if matches else "–")
        return explanation
    except Exception as e:
        return f"Explanation error: {e}"
