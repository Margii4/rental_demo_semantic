from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import logging
import json
import re

from llm_utils import get_embedding, generate_summary, llm_parse_listing, explain_match
from pinecone_utils import upsert_listings, semantic_search

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("rental_assistant")

st.set_page_config(page_title="Rental Assistant Bot", page_icon="🏠", layout="centered")

#  UI LANGUAGES 
LANG = {
    "English": {
        "select_lang": "🌍 Select language",
        "title": "🏠 Rental Assistant Bot – LLM + Pinecone Semantic Search IT",
        "how_to": "ℹ️ How to use this bot (click to expand/collapse)",
        "how_to_text": """
This assistant helps you find the best rental offers in Italy using AI-powered semantic search.

Step-by-step:
1. Specify your wishes: District, pets, renovation, furnishing, price, and any other requirements.
2. Click "Search listings" — you'll see the most relevant results.
        """,
        "district": "District or Area",
        "pets": "Pets allowed?",
        "pets_options": ["Doesn't matter", "Yes", "No"],
        "renovation": "Renovation/Condition",
        "renovation_options": ["Doesn't matter", "Modern", "Basic", "Designer"],
        "furnished": "Furnished?",
        "furnished_options": ["Doesn't matter", "Yes", "No"],
        "price": "Price (e.g., 1000-1400, max 1300, 1000, flexible...)",
        "additional": "Other wishes (e.g., balcony, elevator, garden...)",
        "search": "🔍 Search listings",
        "restart": "↩️ Restart",
        "warn_no_listings": "No valid listings found in the database.",
        "warn_no_results": "No matching listings found.",
        "success_results": "listings found:",
        "save_csv": "💾 Save as CSV",
        "saved": "Results saved as results.csv.",
        "why_result": "💡 Why this result?"
    },
    "Italiano": {
        "select_lang": "🌍 Seleziona lingua",
        "title": "🏠 Assistente Affitti – LLM + Ricerca Semantica Pinecone IT",
        "how_to": "ℹ️ Come usare il bot (clicca per espandere)",
        "how_to_text": """
Questo assistente ti aiuta a trovare le migliori offerte in affitto in Italia usando la ricerca semantica AI.

Passaggi:
1. Specifica le tue preferenze: Zona, animali, stato/ristrutturazione, arredamento, prezzo, altri desideri.
2. Clicca su “Cerca annunci” — vedrai le opzioni più rilevanti.
        """,
        "district": "Zona o Quartiere",
        "pets": "Animali ammessi?",
        "pets_options": ["Non importa", "Sì", "No"],
        "renovation": "Stato/Ristrutturazione",
        "renovation_options": ["Non importa", "Moderno", "Base", "Designer"],
        "furnished": "Arredato?",
        "furnished_options": ["Non importa", "Sì", "No"],
        "price": "Prezzo (es. 1000-1400, max 1200, flessibile...)",
        "additional": "Altri desideri (es. balcone, ascensore, giardino...)",
        "search": "🔍 Cerca annunci",
        "restart": "↩️ Riavvia",
        "warn_no_listings": "Nessun annuncio valido trovato nel database.",
        "warn_no_results": "Nessun annuncio corrispondente trovato.",
        "success_results": "annunci trovati:",
        "save_csv": "💾 Salva come CSV",
        "saved": "Risultati salvati come results.csv.",
        "why_result": "💡 Perché questo risultato?"
    }
}

# SESSION STATE FOR LANGUAGE 
if "lang" not in st.session_state:
    st.session_state["lang"] = "English"

lang = st.radio(
    LANG["English"]["select_lang"],
    list(LANG.keys()),
    horizontal=True,
    index=list(LANG.keys()).index(st.session_state["lang"]),
    key="main_language_radio"
)
st.session_state["lang"] = lang
tr = LANG[lang]

st.title(tr["title"])
with st.expander(tr["how_to"], expanded=True):
    st.markdown(tr["how_to_text"])

# LOAD JSON LISTINGS 
with open("parsers/my_listings.json", "r", encoding="utf-8") as f:
    all_listings = json.load(f)

def filter_by_price(price_filter, price_str):
    """Supports ranges: 1000-1400, max 1300, 1200, flexible (True if match)"""
    if not price_filter or not price_str:
        return True
    price_val = None
    m = re.search(r'(\d+)', str(price_str).replace(",", ""))
    if m:
        price_val = int(m.group(1))
    if "-" in price_filter:
        try:
            min_p, max_p = [int(x.strip()) for x in price_filter.split("-", 1)]
            if price_val is None:
                return False
            return min_p <= price_val <= max_p
        except Exception:
            return False
    if price_filter.strip().lower().startswith("max"):
        try:
            max_p = int(re.findall(r'\d+', price_filter)[0])
            if price_val is None:
                return False
            return price_val <= max_p
        except Exception:
            return False
    try:
        just_num = int(price_filter.strip())
        if price_val is None:
            return False
        return price_val == just_num
    except Exception:
        pass
    return True

# SESSION STATE FOR RESULTS AND FILTERS 
if "results" not in st.session_state:
    st.session_state["results"] = None
if "filters" not in st.session_state:
    st.session_state["filters"] = {}

def clear_all():
    st.session_state["results"] = None
    st.session_state["filters"] = {}

# UI: FILTER FORM 
with st.form("search_form"):
    left_col, right_col = st.columns([1,2])
    with left_col:
        pets_radio = st.radio(
            tr["pets"], tr["pets_options"], horizontal=False,
            index=tr["pets_options"].index(st.session_state["filters"].get("pets", tr["pets_options"][0]))
        )
        renovation_radio = st.radio(
            tr["renovation"], tr["renovation_options"], horizontal=False,
            index=tr["renovation_options"].index(st.session_state["filters"].get("renovation", tr["renovation_options"][0]))
        )
        furnished_radio = st.radio(
            tr["furnished"], tr["furnished_options"], horizontal=False,
            index=tr["furnished_options"].index(st.session_state["filters"].get("furnished", tr["furnished_options"][0]))
        )
    with right_col:
        district = st.text_input(tr["district"], value=st.session_state["filters"].get("district", ""))
        price = st.text_input(tr["price"], value=st.session_state["filters"].get("price", ""))
        additional = st.text_input(tr["additional"], value=st.session_state["filters"].get("additional", ""))
    col_submit, col_restart = st.columns([1, 1])
    with col_submit:
        submitted = st.form_submit_button(tr["search"])
    with col_restart:
        restart = st.form_submit_button(tr["restart"])

if restart:
    clear_all()
    st.rerun()

if submitted:
    # Save filters
    st.session_state["filters"] = {
        "district": district,
        "renovation": renovation_radio,
        "furnished": furnished_radio,
        "pets": pets_radio,
        "price": price,
        "additional": additional
    }
    # Compile query
    query_parts = []
    if district: query_parts.append(f"district: {district}")
    if pets_radio == tr["pets_options"][1]:
        query_parts.append("pets allowed")
    elif pets_radio == tr["pets_options"][2]:
        query_parts.append("no pets")
    if renovation_radio in tr["renovation_options"][1:]:
        query_parts.append(f"renovation: {renovation_radio.lower()}")
    if furnished_radio == tr["furnished_options"][1]:
        query_parts.append("furnished")
    elif furnished_radio == tr["furnished_options"][2]:
        query_parts.append("not furnished")
    if price:
        query_parts.append(f"price: {price}")
    if additional:
        query_parts.append(f"{additional}")
    query = ". ".join(query_parts)
    logger.info(f"Search query: {query}")

    # Local Filtering (speed)
    filtered_listings = []
    for data in all_listings:
        if district and district.lower() not in (str(data.get('district', '')) + str(data.get('description', '')) + str(data.get('title', ''))).lower():
            continue
        if pets_radio == tr["pets_options"][1] and not data.get("pets_allowed", False):
            continue
        if pets_radio == tr["pets_options"][2] and data.get("pets_allowed", True):
            continue
        if renovation_radio != tr["renovation_options"][0] and renovation_radio.lower() not in str(data.get('description', '')).lower():
            continue
        if furnished_radio == tr["furnished_options"][1] and not data.get("furnished", False):
            continue
        if furnished_radio == tr["furnished_options"][2] and data.get("furnished", True):
            continue
        if not filter_by_price(price, data.get('price', '')):
            continue
        if additional and additional.lower() not in str(data.get('description', '')).lower():
            continue
        filtered_listings.append(data)

    if not filtered_listings:
        st.warning(tr["warn_no_listings"])
        st.session_state["results"] = []
    else:
        # Enrich Listings (LLM + Embedding)
        listings = []
        for data in filtered_listings:
            full_desc = f"{data.get('description','')}\nPrice: {data.get('price','')}\nFurnished: {data.get('furnished','')}"
            if "pets_allowed" in data:
                full_desc += f"\nPets allowed: {data['pets_allowed']}"
            llm_fields_raw = llm_parse_listing(full_desc, lang=lang)
            try:
                llm_fields = json.loads(llm_fields_raw)
            except Exception as e:
                logger.error(f"Failed to parse llm_fields for {data.get('url')}: {e}. Raw: {llm_fields_raw}")
                llm_fields = {}
            summary = generate_summary(full_desc, lang=lang)
            data["id"] = data.get("url")
            try:
                embedding = get_embedding(full_desc)
            except Exception as e:
                logger.error(f"Failed to get embedding for {data.get('url')}: {e}")
                embedding = []
            listings.append({**data, **llm_fields, "summary": summary, "embedding": embedding})

        logger.info(f"Final listings for upsert: {listings}")

        # Semantic Search
        query_embedding = get_embedding(query)
        upsert_listings(listings)
        results = semantic_search(query_embedding, top_k=5)
        st.session_state["results"] = results

#  RESULTS DISPLAY 
if st.session_state.get("results") is not None:
    results = st.session_state["results"]
    if not results:
        st.warning(tr["warn_no_results"])
    else:
        st.success(f"{len(results)} {tr['success_results']}")
        for r in results:
            md = r.get("metadata", {})
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except Exception:
                    md = {}
            with st.container():
                st.markdown(
                    f"""**[{md.get('title','(No title)')}]({md.get('url','#')})**  
**Price:** {md.get('price','')}  
**District:** {md.get('district','')}  
**Pets allowed:** {md.get('pets_allowed','–')}  
**Furnished:** {md.get('furnished','–')}  
**Description:** {md.get('description','')}"""
                )
                with st.expander(tr["why_result"], expanded=False):
                    st.info(explain_match(" ".join([str(v) for v in st.session_state["filters"].values()]), md, lang=lang, mode="llm"))

        if st.button(tr["save_csv"], key="save_csv_btn"):
            import pandas as pd
            pd.DataFrame([r.get("metadata", {}) if isinstance(r.get("metadata", {}), dict)
                          else json.loads(r.get("metadata", "{}")) for r in results]).to_csv("results.csv", index=False)
            st.info(tr["saved"])
