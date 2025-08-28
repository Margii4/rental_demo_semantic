from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import logging
import json
import re

from llm_utils import get_embedding, generate_summary, llm_parse_listing, explain_match
from pinecone_utils import semantic_search

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("rental_assistant")

st.set_page_config(page_title="Rental Assistant Bot", page_icon="🏠", layout="centered")

LANG = {
    "English": {
        "select_lang": "🌍 Select language",
        "title": "🏠 Rental Assistant Bot – LLM + Pinecone Semantic Search IT",
        "how_to": "ℹ️ How to use this bot (click to expand/collapse)",
        "how_to_text": """
This assistant helps you find the best rental offers in Italy using AI-powered semantic search.

Step-by-step:
1. Specify your wishes: District, pets, furnishing, price, and any other requirements.
2. Click "Search listings" — you'll see the most relevant results.
        """,
        "district": "District or Area",
        "pets": "Pets allowed?",
        "pets_options": ["Doesn't matter", "Yes", "No"],
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
1. Specifica le tue preferenze: Zona, animali, arredamento, prezzo, altri desideri.
2. Clicca su “Cerca annunci” — vedrai le opzioni più rilevanti.
        """,
        "district": "Zona o Quartiere",
        "pets": "Animali ammessi?",
        "pets_options": ["Non importa", "Sì", "No"],
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

DEFAULT_FILTERS = {
    "district": "",
    "furnished": LANG["English"]["furnished_options"][0],
    "pets": LANG["English"]["pets_options"][0],
    "price": "",
    "additional": ""
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "English"

if "filters" not in st.session_state:
    st.session_state["filters"] = DEFAULT_FILTERS.copy()

if "results" not in st.session_state:
    st.session_state["results"] = None

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

def filter_by_price(price_filter, price_str):
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

def clear_all():
    st.session_state["results"] = None
    st.session_state["filters"] = DEFAULT_FILTERS.copy()
    st.rerun()

def get_pinecone_filters(district, pets_radio, furnished_radio, tr):
    filters = {}
    if district:
        filters["district"] = {"$eq": district}
    if pets_radio == tr["pets_options"][1]:
        filters["pets_allowed"] = {"$eq": True}
    if pets_radio == tr["pets_options"][2]:
        filters["pets_allowed"] = {"$eq": False}
    if furnished_radio == tr["furnished_options"][1]:
        filters["furnished"] = {"$eq": True}
    if furnished_radio == tr["furnished_options"][2]:
        filters["furnished"] = {"$eq": False}
    return filters

with st.form("search_form"):
    left_col, right_col = st.columns([1,2])
    with left_col:
        pets_radio = st.radio(
            tr["pets"], tr["pets_options"], horizontal=False,
            index=tr["pets_options"].index(st.session_state["filters"].get("pets", tr["pets_options"][0])),
            key="pets_radio"
        )
        furnished_radio = st.radio(
            tr["furnished"], tr["furnished_options"], horizontal=False,
            index=tr["furnished_options"].index(st.session_state["filters"].get("furnished", tr["furnished_options"][0])),
            key="furnished_radio"
        )
    with right_col:
        district = st.text_input(tr["district"], value=st.session_state["filters"].get("district", ""), key="district_input")
        price = st.text_input(tr["price"], value=st.session_state["filters"].get("price", ""), key="price_input")
        additional = st.text_input(tr["additional"], value=st.session_state["filters"].get("additional", ""), key="additional_input")
    col_submit, col_restart = st.columns([1, 1])
    with col_submit:
        submitted = st.form_submit_button(tr["search"])
    with col_restart:
        restart = st.form_submit_button(tr["restart"], on_click=clear_all)

if submitted:
    st.session_state["filters"] = {
        "district": district,
        "furnished": furnished_radio,
        "pets": pets_radio,
        "price": price,
        "additional": additional
    }

    query = " ".join([
        f"district: {district}" if district else "",
        f"pets allowed" if pets_radio == tr["pets_options"][1] else "",
        f"no pets" if pets_radio == tr["pets_options"][2] else "",
        f"furnished" if furnished_radio == tr["furnished_options"][1] else "",
        f"not furnished" if furnished_radio == tr["furnished_options"][2] else "",
        f"price: {price}" if price else "",
        f"{additional}" if additional else "",
    ])
    pinecone_filters = get_pinecone_filters(district, pets_radio, furnished_radio, tr)
    query_embedding = get_embedding(query)
    results = semantic_search(query_embedding, top_k=10, filters=pinecone_filters)


    filtered_results = []
    for r in results:
        md = r.get("metadata", {})
     
        if price and not filter_by_price(price, md.get('price', '')):
            continue
     
        if additional and additional.lower() not in (str(md.get('description', '')) + " " + str(md.get('title', ''))).lower():
            continue
        filtered_results.append(r)
    if not filtered_results:
        st.warning(tr["warn_no_listings"])
        st.session_state["results"] = []
    else:
        st.session_state["results"] = filtered_results

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
