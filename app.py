"""
One-Day Presale Development Ingestor – Streamlit MVP

What it does (end to end):
- Lets you add multiple "Developments" (rows)
- For each development, paste URLs (homepage + subpages), optionally auto-discover subpages, and upload local files (PDF/DOCX/TXT)
- Choose the fields you want extracted (or use defaults)
- On submit, scrapes pages, parses documents, and asks an LLM to map the content into your schema
- Shows a results table and lets you download CSV/XLSX

How to run:
1) Save this file as `app.py`
2) Create a virtual env (optional) and install deps:
   pip install streamlit requests beautifulsoup4 trafilatura pdfplumber python-docx pandas openai
3) Set your OpenAI key in your shell (or via Streamlit sidebar):
   export OPENAI_API_KEY=sk-...
4) Run the app:
   streamlit run app.py

Notes:
- This MVP uses basic requests/bs4 scraping (fast). If a site needs JS rendering, add Playwright later.
- LLM extraction is schema-driven using the fields you provide.
- Keep usage within the target domain’s Terms of Use and robots.txt.
"""

import os
import io
import re
import time
import json
import hashlib
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import trafilatura
import pdfplumber
from docx import Document as DocxDocument

# Optional: If you plan to use OpenAI's new SDK name/shape, adjust here.
try:
    from openai import OpenAI  # modern SDK
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ------------------------------
# Defaults & Helpers
# ------------------------------
DEFAULT_FIELDS = [
    "Project Name",
    "Developer Name",
    "Address",
    "City",
    "Neighbourhood",
    "Home Type",
    "Starting Price",
    "Price Range",
    "Unit Mix",
    "Square Footage Range",
    "Number of Floors",
    "Number of Units",
    "Completion (Year/Quarter)",
    "Deposit Structure",
    "Sales Start Date",
    "Incentives / Promotions",
    "Amenities",
    "Website",
    "Brochure Link",
]

SESSION_KEY = "dev_rows"
HEADERS = {
    "User-Agent": "ListingsNearbyBot/1.0 (+https://listingsnearby.com)"
}

# ------------------------------
# Scraping & Parsing
# ------------------------------

def normalize_url(base_url: str, href: str) -> str:
    from urllib.parse import urljoin
    return urljoin(base_url, href)


def same_domain(base_url: str, candidate: str) -> bool:
    from urllib.parse import urlparse
    a = urlparse(base_url)
    b = urlparse(candidate)
    return (a.netloc.lower() == b.netloc.lower()) and (b.scheme in ("http", "https"))


def fetch_html(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return ""


def extract_text_from_html(html: str, url: str = "") -> str:
    # Try trafilatura first
    try:
        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if extracted and len(extracted) > 50:
            return extracted
    except Exception:
        pass
    # Fallback: bs4 get_text
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text.strip()
    except Exception:
        return ""


def discover_subpages(base_url: str, html: str, max_pages: int = 15) -> List[str]:
    """Collect internal links that look relevant (pricing, floorplans, features, deposit, etc.)."""
    tokens = ["pricing", "floorplan", "plans", "amenities", "features", "deposit", "incentive", "register", "neighbourhood", "location", "brochure", "downloads", "spec", "faq"]
    urls = set()
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        abs_url = normalize_url(base_url, href)
        if same_domain(base_url, abs_url):
            path = abs_url.lower()
            if any(tok in path for tok in tokens):
                urls.add(abs_url)
    # If too few found, add all internal links up to cap
    if len(urls) < 3:
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            abs_url = normalize_url(base_url, href)
            if same_domain(base_url, abs_url):
                urls.add(abs_url)
                if len(urls) >= max_pages:
                    break
    return list(urls)[:max_pages]


def read_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            texts = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    texts.append(t)
            return "\n\n".join(texts)
    except Exception:
        return ""


def read_docx(file_bytes: bytes) -> str:
    try:
        bio = io.BytesIO(file_bytes)
        doc = DocxDocument(bio)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ------------------------------
# LLM Extraction
# ------------------------------

LLM_MODEL = "gpt-4o-mini"  # Adjust to your available model


def get_openai_client_from_env(api_key: str | None = None):
    if not _HAS_OPENAI:
        return None
    key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        client = OpenAI(api_key=key)
        return client
    except Exception:
        return None


def llm_extract_schema(client, fields: List[str], text_chunk: str) -> Dict[str, Any]:
    """Ask the LLM to return a JSON object with the requested fields."""
    system_prompt = (
        "You are a data extraction engine for Canadian presale real estate projects. "
        "Return ONLY a strict JSON object with the requested fields as keys. "
        "If a field is not found, return an empty string for that field. "
        "Normalize values (e.g., dates as 'YYYY' or 'YYYY Q#', price as '$123,456')."
    )

    user_prompt = (
        "Extract the following fields from the text.\n\n"
        f"FIELDS: {fields}\n\n"
        "TEXT:\n" + text_chunk[:120000]
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        # Ensure all fields exist
        for f in fields:
            data.setdefault(f, "")
        return data
    except Exception as e:
        # On any failure, return empty dict with keys
        return {f: "" for f in fields}


def aggregate_fieldwise(results: List[Dict[str, Any]], fields: List[str]) -> Dict[str, Any]:
    """Combine chunk-level results into one per development.
    Strategy: choose the longest non-empty value per field; fallback empty.
    """
    final = {f: "" for f in fields}
    for f in fields:
        candidates = [r.get(f, "") for r in results if r.get(f, "").strip()]
        if candidates:
            # Choose longest (often most complete)
            final[f] = max(candidates, key=lambda x: len(x))
    return final


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Presale Dev Ingestor (1‑Day MVP)", layout="wide")

st.title("🏗️ Presale Development Ingestor – 1‑Day MVP")
st.write(
    "Paste URLs & upload docs per development → pick fields → get a clean table (CSV/XLSX)."
)

with st.sidebar:
    st.header("Settings")
    st.caption("Optional: set your OpenAI API key here if not using env var.")
    manual_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    request_delay = st.slider("Delay between requests (seconds)", 0.0, 3.0, 0.5, 0.1)
    llm_on = st.toggle("Use LLM extraction", value=True)
    max_auto_pages = st.slider("Max auto-discovered subpages", 3, 30, 15, 1)

if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = []

colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Fields to Extract")
    fields_text = st.text_area(
        "Comma-separated field names",
        value=", ".join(DEFAULT_FIELDS),
        height=100,
    )
    fields = [f.strip() for f in fields_text.split(",") if f.strip()]

with colB:
    st.subheader("Add Developments")
    if st.button("➕ Add Development Row", use_container_width=True):
        st.session_state[SESSION_KEY].append({
            "name": "",
            "urls": "",
            "auto": True,
            "files": [],
        })

# Render rows
rows = st.session_state[SESSION_KEY]
if not rows:
    st.info("Click **Add Development Row** to begin.")

for idx, row in enumerate(rows):
    with st.expander(f"Development #{idx+1}", expanded=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            row["name"] = st.text_input("Development Name", value=row.get("name", ""), key=f"name_{idx}")
            row["auto"] = st.checkbox("Auto-discover subpages from first URL", value=row.get("auto", True), key=f"auto_{idx}")
            row["urls"] = st.text_area(
                "URLs (one per line)",
                value=row.get("urls", ""),
                height=120,
                key=f"urls_{idx}"
            )
        with c2:
            uploaded = st.file_uploader(
                "Upload PDFs / DOCX / TXT (multiple)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key=f"files_{idx}"
            )
            if uploaded:
                row["files"] = uploaded
        # Delete row button
        st.button("🗑️ Remove this row", key=f"del_{idx}", on_click=lambda i=idx: st.session_state[SESSION_KEY].pop(i))

st.divider()

run = st.button("🚀 Extract Data", type="primary", use_container_width=True)

results_table: List[Dict[str, Any]] = []
progress = st.empty()

if run:
    client = get_openai_client_from_env(manual_key if manual_key else None)
    if llm_on and client is None:
        st.warning("LLM extraction is ON but no valid OpenAI key found. Turn off LLM or set a key.")

    for i, row in enumerate(st.session_state[SESSION_KEY]):
        dev_name = row.get("name", "").strip() or f"Development {i+1}"
        progress.markdown(f"**Processing:** {dev_name} ({i+1}/{len(rows)})")

        # 1) Gather URLs (manual + auto)
        input_urls = [u.strip() for u in (row.get("urls") or "").splitlines() if u.strip()]
        all_urls = []
        collected_texts: List[str] = []

        # Fetch primary pages
        for u in input_urls:
            html = fetch_html(u)
            if html:
                txt = extract_text_from_html(html, u)
                if txt:
                    collected_texts.append(txt)
            all_urls.append(u)
            time.sleep(request_delay)

        # Auto-discover subpages from the FIRST URL only (simple + fast)
        if row.get("auto") and input_urls:
            first_url = input_urls[0]
            html = fetch_html(first_url)
            discovered = discover_subpages(first_url, html or "", max_auto_pages)
            # Fetch discovered pages
            for u in discovered:
                if u in all_urls:
                    continue
                html = fetch_html(u)
                if html:
                    txt = extract_text_from_html(html, u)
                    if txt:
                        collected_texts.append(txt)
                all_urls.append(u)
                time.sleep(request_delay)

        # 2) Read uploaded files
        file_links = []
        for f in (row.get("files") or []):
            data = f.read()
            name_lower = f.name.lower()
            text = ""
            if name_lower.endswith(".pdf"):
                text = read_pdf(data)
            elif name_lower.endswith(".docx"):
                text = read_docx(data)
            elif name_lower.endswith(".txt"):
                text = read_txt(data)
            if text:
                collected_texts.append(text)
                # Not a real URL, but we can store file names as brochure links if desired
                if name_lower.endswith(".pdf") and "brochure" in name_lower:
                    file_links.append(f.name)

        # 3) LLM extraction per chunk
        # Simple chunking by characters to stay under token limits
        big_text = "\n\n".join(collected_texts)
        chunks = []
        MAX_CHARS = 20000  # approx; adjust if needed
        for start in range(0, len(big_text), MAX_CHARS):
            chunks.append(big_text[start:start+MAX_CHARS])

        per_chunk_results: List[Dict[str, Any]] = []
        if llm_on and client:
            for ch in chunks if chunks else [""]:
                if not ch.strip():
                    continue
                data = llm_extract_schema(client, fields, ch)
                per_chunk_results.append(data)
        else:
            # If no LLM, return empties and rely on a couple regex heuristics
            heuristic = {f: "" for f in fields}
            # Basic heuristics for a few fields
            text0 = big_text
            # price
            m = re.search(r"\$\s?([0-9][0-9,]{2,})", text0)
            if m and "Starting Price" in fields:
                heuristic["Starting Price"] = f"${m.group(1)}"
            # completion year
            m2 = re.search(r"(20[2-4][0-9])\s*(?:Q([1-4]))?", text0)
            if m2 and "Completion (Year/Quarter)" in fields:
                heuristic["Completion (Year/Quarter)"] = m2.group(1) + (f" Q{m2.group(2)}" if m2.group(2) else "")
            # deposit structure
            if "Deposit Structure" in fields:
                m3 = re.search(r"deposit[^\n]*:?[\s\-]*([\s\S]{0,120})", text0, re.IGNORECASE)
                if m3:
                    heuristic["Deposit Structure"] = m3.group(1).strip()
            per_chunk_results.append(heuristic)

        # 4) Aggregate fields
        agg = aggregate_fieldwise(per_chunk_results, fields)

        # Fill convenience extras
        if "Project Name" in agg and not agg.get("Project Name"):
            agg["Project Name"] = dev_name
        if "Website" in agg and not agg.get("Website") and input_urls:
            agg["Website"] = input_urls[0]
        if "Brochure Link" in agg and not agg.get("Brochure Link") and file_links:
            agg["Brochure Link"] = ", ".join(file_links)

        results_table.append(agg)

    progress.empty()

if results_table:
    st.success("Extraction complete.")
    df = pd.DataFrame(results_table)
    st.dataframe(df, use_container_width=True)

    # Downloads
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="presale_developments.csv", mime="text/csv")

    # XLSX
    xls_io = io.BytesIO()
    with pd.ExcelWriter(xls_io, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Developments")
    st.download_button("⬇️ Download Excel", data=xls_io.getvalue(), file_name="presale_developments.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("MVP limitations: Some JS-rendered pages may need Playwright; heavy PDFs may extract imperfectly; adjust chunk size/model as needed.")
