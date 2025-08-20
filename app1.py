# Streamlit RAG Chat — VERINT-branded (centered header, transcripts, blue/white theme)
# Uses FAISS (text+image), OpenAI embeddings, cross-encoder reranker, and Sources view.
# This version reads indices & metadata from ./cloud_data/ (repo folder) for Streamlit Cloud.

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import os
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# =======================
# BRAND / THEME
# =======================
BRAND_BLUE = "#0078FF"      # VERINT blue
DEEP_BLUE  = "#0B3D91"      # readable deep blue

st.set_page_config(page_title="VERINT RAG", page_icon="💬", layout="wide")

# Global CSS (white bg, blue text, centered header, brighter sidebar headings)
st.markdown(f"""
<style>
    .stApp {{
        background-color: #FFFFFF;
        color: {DEEP_BLUE};
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {{
        color: {DEEP_BLUE} !important;
    }}
    .verint-header {{
        width: 100%;
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: {BRAND_BLUE};
        margin: 0.25rem 0 1rem 0;
    }}
    section[data-testid="stSidebar"] h2 {{
        color: {BRAND_BLUE} !important;
    }}
    .download-transcript {{
        color: {BRAND_BLUE};
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }}
    a, a:visited {{
        color: {BRAND_BLUE} !important;
        text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)

# =======================
# ENV / PATHS (Cloud-first)
# =======================
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")  # optional (local dev)

# Allow key from .env OR Streamlit secrets
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
)
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to **.env** or **Streamlit secrets**.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Cloud data folder (committed to repo) ----
CLOUD_DIR = PROJECT_ROOT / "cloud_data"

def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None

# Standard filenames we asked you to upload
TEXT_INDEX_PATH = CLOUD_DIR / "faiss_text.index"
IMAGE_INDEX_PATH = CLOUD_DIR / "faiss_image.index"       # optional
TEXT_META_CSV   = _first_existing(
    CLOUD_DIR / "text_metadata.csv", CLOUD_DIR / "text_meta.csv"
)
IMG_META_CSV    = _first_existing(
    CLOUD_DIR / "image_metadata.csv", CLOUD_DIR / "img_meta.csv"
)  # optional
TEXT_EMB_NPY    = _first_existing(
    CLOUD_DIR / "text_embeddings.npy",
)
TEXT_CHUNKS_JSONL = _first_existing(
    CLOUD_DIR / "text_chunks.jsonl",
)

# Quick sanity display
with st.sidebar:
    st.caption("**Data folder:** `./cloud_data/`")
    st.code(str(CLOUD_DIR), language="bash")

# =======================
# MODELS
# =======================
TEXT_EMBED_MODEL = "text-embedding-3-small"                # OpenAI
CHAT_MODEL       = "gpt-4o-mini"                           # OpenAI
CLIP_TEXT_MODEL  = "sentence-transformers/clip-ViT-B-32"   # sentence-transformers
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # sentence-transformers

# =======================
# HEADER (Text only)
# =======================
st.markdown('<div class="verint-header">VERINT</div>', unsafe_allow_html=True)

# =======================
# HELPERS
# =======================
def l2norm(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        v = v[None, :]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return (v / n).astype("float32")

def _require_exists(path: Path, label: str) -> None:
    if not path or not path.exists():
        st.error(f"Required file for **{label}** is missing: `{path}`")
        st.info(
            "Make sure it is uploaded to the `cloud_data/` folder in your GitHub repo. "
            "Then click **Rerun** in Streamlit."
        )
        st.stop()

# =======================
# LOAD ALL (cached)
# =======================
@st.cache_resource(show_spinner=False)
def load_all():
    # Check mandatory files
    _require_exists(TEXT_INDEX_PATH, "FAISS text index")
    _require_exists(TEXT_META_CSV, "text metadata CSV")
    _require_exists(TEXT_EMB_NPY, "text embeddings (.npy)")

    try:
        text_index = faiss.read_index(str(TEXT_INDEX_PATH))
    except Exception as e:
        st.error(f"Failed to read FAISS text index: `{TEXT_INDEX_PATH}`\n\n{e}")
        st.stop()

    img_index = None
    if IMAGE_INDEX_PATH and IMAGE_INDEX_PATH.exists():
        try:
            img_index = faiss.read_index(str(IMAGE_INDEX_PATH))
        except Exception as e:
            st.warning(f"Image index present but failed to load: {e}")

    try:
        text_meta = pd.read_csv(TEXT_META_CSV)
    except Exception as e:
        st.error(f"Failed to read text metadata: `{TEXT_META_CSV}`\n\n{e}")
        st.stop()

    img_meta = pd.DataFrame()
    if IMG_META_CSV and IMG_META_CSV.exists():
        try:
            img_meta = pd.read_csv(IMG_META_CSV)
        except Exception as e:
            st.warning(f"Failed to read image metadata CSV: {e}")

    try:
        text_emb = np.load(TEXT_EMB_NPY).astype("float32")
    except Exception as e:
        st.error(f"Failed to read text embeddings: `{TEXT_EMB_NPY}`\n\n{e}")
        st.stop()

    # chunk_id -> full JSON record (to fetch chunk text)
    chunk_by_id: Dict[str, Dict[str, Any]] = {}
    if TEXT_CHUNKS_JSONL and TEXT_CHUNKS_JSONL.exists():
        with open(TEXT_CHUNKS_JSONL, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cid = rec.get("chunk_id")
                    if cid:
                        chunk_by_id[cid] = rec
                except Exception:
                    pass

    return text_index, img_index, text_meta, img_meta, text_emb, chunk_by_id

text_index, img_index, text_meta, img_meta, TEXT_EMB, chunk_by_id = load_all()

@st.cache_resource(show_spinner=False)
def load_clip():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CLIP_TEXT_MODEL, device="cpu")

def embed_query_openai(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=TEXT_EMBED_MODEL, input=[q])
    return l2norm(np.array(resp.data[0].embedding, "float32")[None, :])

def faiss_search(index: faiss.Index, q: np.ndarray, k: int):
    D, I = index.search(q, k)
    return D[0], I[0]

def mmr_select(q: np.ndarray, cand_ids: List[int], cand_vecs: np.ndarray,
               k: int, lambda_div: float) -> List[int]:
    if not cand_ids:
        return []
    k = min(k, len(cand_ids))
    q = q.astype("float32")
    sim_to_q = (cand_vecs @ q.T).ravel()
    sel, remaining = [], list(range(len(cand_ids)))
    first = int(np.argmax(sim_to_q))
    sel.append(first); remaining.remove(first)
    cc_sim = cand_vecs @ cand_vecs.T
    while len(sel) < k and remaining:
        best, bestscore = None, -1e9
        for j in remaining:
            div = np.max(cc_sim[j, sel])
            score = lambda_div * sim_to_q[j] - (1.0 - lambda_div) * div
            if score > bestscore:
                bestscore, best = score, j
        sel.append(best); remaining.remove(best)
    return sel

def retrieve(query: str, k_text=8, k_img=4, mmr_pool=50, lambda_div=0.55, per_doc_cap=2, use_images=True):
    # Text pool
    q_openai = embed_query_openai(query)
    tD, tI = faiss_search(text_index, q_openai, max(k_text, mmr_pool))
    pool = [(int(i), float(s)) for i, s in zip(tI, tD) if i != -1 and i < len(text_meta)]
    results: List[Dict[str, Any]] = []

    if pool:
        cand_ids = [i for i, _ in pool]
        cand_vecs = TEXT_EMB[np.array(cand_ids)]
        selpos = mmr_select(q_openai, cand_ids, cand_vecs, k=k_text, lambda_div=lambda_div)
        picked = [pool[p] for p in selpos]

        per_doc: Dict[str, int] = {}
        for idx, sc in picked:
            row = text_meta.iloc[idx].to_dict()
            fn = row.get("file_name", "")
            if per_doc.get(fn, 0) >= per_doc_cap:
                continue
            per_doc[fn] = per_doc.get(fn, 0) + 1
            row.update({"_score": float(sc), "_modality": "text", "_faiss_id": int(idx)})
            results.append(row)
            if len(results) >= k_text:
                break

    # Optional image hits via CLIP
    if use_images and img_index is not None and img_index.ntotal > 0 and not img_meta.empty and k_img > 0:
        clip = load_clip()
        q_clip = clip.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        iD, iI = faiss_search(img_index, q_clip, k_img)
        for idx, sc in zip(iI, iD):
            if idx == -1 or idx >= len(img_meta):
                continue
            row = img_meta.iloc[int(idx)].to_dict()
            row.update({"_score": float(sc), "_modality": "image", "_faiss_id": int(idx)})
            results.append(row)

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results

def rerank_text_hits(query: str, text_hits: List[Dict[str, Any]], use_rerank=True):
    if not use_rerank or not text_hits:
        return text_hits
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    pairs = []
    for h in text_hits:
        cid = h.get("chunk_id")
        txt = (chunk_by_id.get(cid) or {}).get("text") or \
              f"{h.get('file_name','')} pages {h.get('page_start','?')}-{h.get('page_end','?')} {h.get('heading_path','')}"
        pairs.append([query, txt])
    scores = reranker.predict(pairs, show_progress_bar=False)
    for h, s in zip(text_hits, scores):
        h["_rerank"] = float(s)
    text_hits.sort(key=lambda x: x.get("_rerank", 0.0), reverse=True)
    return text_hits

def build_context(text_hits: List[Dict[str, Any]], char_limit=4500):
    pieces, kept = [], []
    for h in text_hits:
        cid = h.get("chunk_id")
        txt = (chunk_by_id.get(cid) or {}).get("text", "")
        if not txt:
            continue
        if sum(len(p) for p in pieces) + len(txt) > char_limit:
            break
        pieces.append(txt)
        kept.append(h)
    return ("\n\n---\n\n".join(pieces), kept)

def answer(query: str, context: str) -> str:
    sys = ("You are a helpful assistant that answers strictly from the provided context. "
           "If the context does not contain the answer, say you don't know. "
           "Cite filenames and page ranges when useful.")
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return resp.choices[0].message.content.strip()

# =======================
# TRANSCRIPT HELPERS
# =======================
def transcript_as_markdown(history: List[Dict[str, Any]]) -> str:
    lines = ["# VERINT RAG Chat Transcript", ""]
    for turn in history:
        who = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"## {who}")
        lines.append(turn["content"])
        lines.append("")
        if turn.get("sources"):
            lines.append("**Sources:**")
            for i, h in enumerate(turn["sources"], 1):
                fn = h.get("file_name","?")
                ps = f"{h.get('page_start','?')}-{h.get('page_end','?')}"
                hp = h.get("heading_path","")
                lines.append(f"- {i}. `{fn}` — pages {ps}  {('• '+hp) if hp else ''}")
            lines.append("")
    return "\n".join(lines)

def transcript_as_json(history: List[Dict[str, Any]]) -> bytes:
    return json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8")

# =======================
# SIDEBAR
# =======================
with st.sidebar:
    st.subheader("Retrieval Settings")
    k_text = st.slider("Text results (k)", 4, 20, 8, 1)
    k_img  = st.slider("Image results", 0, 8, 4, 1)
    mmr_pool = st.slider("MMR pool size", 20, 120, 50, 5)
    lambda_div = st.slider("MMR λ (relevance↔diversity)", 0.1, 0.9, 0.55, 0.05)
    per_doc_cap = st.slider("Max chunks per file", 1, 4, 2, 1)
    use_images  = st.checkbox("Use image index", True)
    use_rerank  = st.checkbox("Cross‑encoder rerank", True, help="Improves precision; slightly slower")

    st.markdown('<div class="download-transcript">Download transcript</div>', unsafe_allow_html=True)
    if "history" not in st.session_state:
        st.session_state.history = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_name   = f"verint_rag_transcript_{ts}.md"
    json_name = f"verint_rag_transcript_{ts}.json"

    md_bytes = transcript_as_markdown(st.session_state.history).encode("utf-8")
    st.download_button("⬇️ Markdown", md_bytes, file_name=md_name, mime="text/markdown")
    st.download_button("⬇️ JSON", transcript_as_json(st.session_state.history), file_name=json_name, mime="application/json")

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# =======================
# CHAT
# =======================
if "history" not in st.session_state:
    st.session_state.history = []

# Render history
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources", expanded=False):
                for i, h in enumerate(m["sources"], 1):
                    fn = h.get("file_name","?")
                    ps = f"{h.get('page_start','?')}-{h.get('page_end','?')}"
                    hp = h.get("heading_path","")
                    st.markdown(f"**{i}.** `{fn}` — pages {ps}  {('• '+hp) if hp else ''}")

# Input
prompt = st.chat_input("Ask a question about your PDFs…")
if prompt:
    st.session_state.history.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            hits = retrieve(prompt, k_text=k_text, k_img=k_img,
                            mmr_pool=mmr_pool, lambda_div=lambda_div,
                            per_doc_cap=per_doc_cap, use_images=use_images)
            text_hits = [h for h in hits if h.get("_modality") == "text"]
            text_hits = rerank_text_hits(prompt, text_hits, use_rerank=use_rerank)
            ctx, used = build_context(text_hits, char_limit=4500)

        if not used:
            out = "_I couldn’t find relevant context._"
            st.markdown(out)
            st.session_state.history.append({"role":"assistant","content":out})
        else:
            with st.spinner("Thinking…"):
                out = answer(prompt, ctx)
            st.markdown(out)
            st.session_state.history.append({"role":"assistant","content":out,"sources":used})

            with st.expander("Sources", expanded=True):
                for i, h in enumerate(used, 1):
                    fn = h.get("file_name","?"); ps = f"{h.get('page_start','?')}-{h.get('page_end','?')}"
                    hp = h.get("heading_path","")
                    st.markdown(f"**{i}.** `{fn}` — pages {ps}  {('• '+hp) if hp else ''}")
