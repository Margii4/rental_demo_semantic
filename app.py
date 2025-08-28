from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import logging
import json
import re

from llm_utils import get_embedding, explain_match
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
        "title": "🏠 Rental Assistant Bot – LLM + Pinecone Semantic Search (IT)",
        "how_to": "ℹ️ How to use this bot (click to expand)",
        "how_to_text": """
This assistant helps you find rental offers in Italy using AI semantic search.

Steps:
1) Set your preferences: Area, pets, furnishing, price, other wishes.
2) Click “Search listings” to see the most relevant results.
        """,
        "district": "District or Area",
        "pets": "Pets allowed?",
        "pets_labels": {"any": "Doesn't matter", "yes": "Yes", "no": "No"},
        "furnished": "Furnished?",
        "furnished_labels": {"any": "Doesn't matter", "yes": "Yes", "no": "No"},
        "price": "Price (e.g., 1000-1400, max 1300, 1000, flexible...)",
        "additional": "Other wishes (e.g., balcony, elevator, garden...)",
        "search": "🔍 Search listings",
        "restart": "↩️ Restart",
        "warn_no_listings": "No valid listings found in the database.",
        "warn_no_results": "No matching listings found.",
        "success_results": "listings found:",
        "save_csv": "💾 Save as CSV",
        "saved": "Results saved as results.csv.",
        "why_result": "💡 Why this result?",
        "price_lbl": "Price",
        "district_lbl": "District",
        "pets_lbl": "Pets allowed",
        "furn_lbl": "Furnished",
        "desc_lbl": "Description",
    },
    "Italiano": {
        "select_lang": "🌍 Seleziona lingua",
        "title": "🏠 Assistente Affitti – Ricerca Semantica (IT)",
        "how_to": "ℹ️ Come usare il bot (clicca per espandere)",
        "how_to_text": """
Questo assistente trova annunci in affitto in Italia con ricerca semantica AI.

Passaggi:
1) Imposta preferenze: Zona, animali, arredamento, prezzo, altri desideri.
2) Clicca “Cerca annunci” per vedere i risultati più rilevanti.
        """,
        "district": "Zona o Quartiere",
        "pets": "Animali ammessi?",
        "pets_labels": {"any": "Non importa", "yes": "Sì", "no": "No"},
        "furnished": "Arredato?",
        "furnished_labels": {"any": "Non importa", "yes": "Sì", "no": "No"},
        "price": "Prezzo (es. 1000-1400, max 1200, flessibile...)",
        "additional": "Altri desideri (es. balcone, ascensore, giardino...)",
        "search": "🔍 Cerca annunci",
        "restart": "↩️ Riavvia",
        "warn_no_listings": "Nessun annuncio valido trovato nel database.",
        "warn_no_results": "Nessun annuncio corrispondente trovato.",
        "success_results": "annunci trovati:",
        "save_csv": "💾 Salva come CSV",
        "saved": "Risultati salvati come results.csv.",
        "why_result": "💡 Perché questo risultato?",
        "price_lbl": "Prezzo",
        "district_lbl": "Zona",
        "pets_lbl": "Animali ammessi",
        "furn_lbl": "Arredato",
        "desc_lbl": "Descrizione",
    }
}

PETS_VALUES = ["any", "yes", "no"]
FURNISHED_VALUES = ["any", "yes", "no"]

DEFAULT_FILTERS = {
    "district": "",
    "furnished": "any",
    "pets": "any",
    "price": "",
    "additional": ""
}


if "lang" not in st.session_state:
    st.session_state["lang"] = "English"
if "filters" not in st.session_state:
    st.session_state["filters"] = DEFAULT_FILTERS.copy()
if "results" not in st.session_state:
    st.session_state["results"] = None

def _normalize_choice(val, is_pets=True):
    v = str(val or "").strip().lower()
   
    mapping = {
        "doesn't matter": "any",
        "doesnt matter": "any",
        "non importa": "any",
        "any": "any",
        "yes": "yes",
        "sì": "yes",
        "si": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
    }
    return mapping.get(v, "any")

for key in ("pets", "furnished"):
    v = st.session_state["filters"].get(key)
    if v not in PETS_VALUES:  # same set for both fields
        st.session_state["filters"][key] = _normalize_choice(v, is_pets=(key=="pets"))

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

def _label(format_dict):
    return lambda v: format_dict.get(v, v)

def filter_by_price(price_filter, price_str):
    if not price_filter or not price_str:
        return True
    price_val = None
    m = re.search(r'(\d+)', str(price_str).replace(",", ""))
    if m:
        price_val = int(m.group(1))

    pf = price_filter.strip().lower()
    if "-" in pf:
        try:
            min_p, max_p = [int(x.strip()) for x in pf.split("-", 1)]
            return price_val is not None and (min_p <= price_val <= max_p)
        except Exception:
            return False
    if pf.startswith("max"):
        try:
            max_p = int(re.findall(r'\d+', pf)[0])
            return price_val is not None and price_val <= max_p
        except Exception:
            return False
    try:
        just_num = int(pf)
        return price_val is not None and price_val == just_num
    except Exception:
        return True  

def clear_all():
    st.session_state["results"] = None
    st.session_state["filters"] = DEFAULT_FILTERS.copy()
    st.rerun()

def get_pinecone_filters(district, pets_value, furnished_value):
    f = {}
    if district:
        f["district"] = {"$eq": district}
    if pets_value == "yes":
        f["pets_allowed"] = {"$eq": True}
    elif pets_value == "no":
        f["pets_allowed"] = {"$eq": False}
    if furnished_value == "yes":
        f["furnished"] = {"$eq": True}
    elif furnished_value == "no":
        f["furnished"] = {"$eq": False}
    return f

with st.form("search_form"):
    left_col, right_col = st.columns([1, 2])

    with left_col:
        pets_value = st.radio(
            tr["pets"],
            options=PETS_VALUES,
            index=PETS_VALUES.index(st.session_state["filters"].get("pets", "any")),
            key="pets_radio",
            format_func=_label(tr["pets_labels"])
        )
        furnished_value = st.radio(
            tr["furnished"],
            options=FURNISHED_VALUES,
            index=FURNISHED_VALUES.index(st.session_state["filters"].get("furnished", "any")),
            key="furnished_radio",
            format_func=_label(tr["furnished_labels"])
        )

    with right_col:
        district = st.text_input(tr["district"], value=st.session_state["filters"].get("district", ""), key="district_input")
        price = st.text_input(tr["price"], value=st.session_state["filters"].get("price", ""), key="price_input")
        additional = st.text_input(tr["additional"], value=st.session_state["filters"].get("additional", ""), key="additional_input")

    c1, c2 = st.columns([1, 1])
    with c1:
        submitted = st.form_submit_button(tr["search"])
    with c2:
        restart = st.form_submit_button(tr["restart"], on_click=clear_all)

if submitted:
    st.session_state["filters"] = {
        "district": district,
        "furnished": furnished_value,
        "pets": pets_value,
        "price": price,
        "additional": additional
    }

    query_parts = [
        f"district: {district}" if district else "",
        "pets allowed" if pets_value == "yes" else ("no pets" if pets_value == "no" else ""),
        "furnished" if furnished_value == "yes" else ("not furnished" if furnished_value == "no" else ""),
        f"price: {price}" if price else "",
        additional or "",
    ]
    query = " ".join([p for p in query_parts if p])

    pinecone_filters = get_pinecone_filters(district, pets_value, furnished_value)
    emb = get_embedding(query)
    results = semantic_search(emb, top_k=10, filters=pinecone_filters)

    filtered_results = []
    for r in results:
        md = r.get("metadata", {})
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except Exception:
                md = {}
        if price and not filter_by_price(price, md.get("price", "")):
            continue
        if additional and additional.lower() not in (str(md.get("description", "")) + " " + str(md.get("title", ""))).lower():
            continue
        filtered_results.append({"score": r.get("score"), "metadata": md})

    st.session_state["results"] = filtered_results if filtered_results else []

if st.session_state.get("results") is not None:
    results = st.session_state["results"]
    if not results:
        st.warning(tr["warn_no_results"])
    else:
        st.success(f"{len(results)} {tr['success_results']}")
        for item in results:
            md = item["metadata"]
            title = md.get("title", "(No title)")
            url = md.get("url", "#")
            st.markdown(
                f"""**[{title}]({url})**  
**{tr['price_lbl']}:** {md.get('price','')}  
**{tr['district_lbl']}:** {md.get('district','')}  
**{tr['pets_lbl']}:** {md.get('pets_allowed','–')}  
**{tr['furn_lbl']}:** {md.get('furnished','–')}  
**{tr['desc_lbl']}:** {md.get('description','')}"""
            )
            with st.expander(tr["why_result"], expanded=False):
                st.info(explain_match(
                    " ".join(str(v) for v in st.session_state["filters"].values()),
                    md,
                    lang=st.session_state["lang"],
                    mode="llm"
                ))

        if st.button(tr["save_csv"], key="save_csv_btn"):
            import pandas as pd
            df = pd.DataFrame([i["metadata"] for i in results])
            df.to_csv("results.csv", index=False)
            st.info(tr["saved"])
