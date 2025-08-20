# Phase 5 — Retrieval + Diversification (MMR) + Rerank + Answer
# Works with Phase 4 outputs (faiss_text.index, text_embeddings.npy, metadata CSVs, text_chunks.jsonl)

# ---- OpenMP safety on macOS (avoids libomp duplicate runtime aborts) ----
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# Load API key from .env in project root
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        f"OPENAI_API_KEY not found.\n"
        f"Expected in: {PROJECT_ROOT / '.env'}\n"
        f"Format: OPENAI_API_KEY=sk-...your_key...\n"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Paths produced by Phase 4
# ---------------------------
cfg = json.loads((PROJECT_ROOT / "config.json").read_text())
P = {k: Path(v).expanduser().resolve() for k, v in cfg["paths"].items()}

INDEX_DIR = P["INDEX_DIR"]
EMBED_DIR = P["EMBED_DIR"]
CHUNK_DIR = P["CHUNK_DIR"]

TEXT_INDEX_PATH = INDEX_DIR / "faiss_text.index"
IMAGE_INDEX_PATH = INDEX_DIR / "faiss_image.index"
TEXT_META_CSV  = EMBED_DIR / "text_metadata.csv"
IMG_META_CSV   = EMBED_DIR / "image_metadata.csv"
TEXT_EMB_NPY   = EMBED_DIR / "text_embeddings.npy"     # rows align with FAISS ids
TEXT_CHUNKS_JSONL = CHUNK_DIR / "text_chunks.jsonl"    # JSONL

# ---------------------------
# Models
# ---------------------------
TEXT_EMBED_MODEL = "text-embedding-3-small"  # 1536-d
CHAT_MODEL       = "gpt-4o-mini"
CLIP_TEXT_MODEL  = "sentence-transformers/clip-ViT-B-32"

# Reranker (cross-encoder)
USE_RERANK = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, fast; CPU OK

# ---------------------------
# Helpers
# ---------------------------
def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype("float32")

def load_faiss(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(str(path))

def load_meta_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_text_chunks_map(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map chunk_id -> full record (to fetch full text for context)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if cid:
                    out[cid] = rec
            except Exception:
                continue
    return out

def embed_query_openai(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=TEXT_EMBED_MODEL, input=[q])
    vec = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    return l2_normalize(vec)

_clip_text_encoder = None
def clip_text_embed(qs: List[str]) -> np.ndarray:
    global _clip_text_encoder
    if _clip_text_encoder is None:
        from sentence_transformers import SentenceTransformer
        _clip_text_encoder = SentenceTransformer(CLIP_TEXT_MODEL, device="cpu")
    v = _clip_text_encoder.encode(qs, batch_size=32, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False).astype("float32")
    return v

def faiss_search(index: faiss.Index, qvec: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(qvec, top_k)  # IP on normalized ≈ cosine
    return scores[0], ids[0]

# ---------------------------
# Diversification (MMR)
# ---------------------------
def mmr_select(
    query_vec: np.ndarray,
    cand_ids: List[int],
    cand_vecs: np.ndarray,
    k: int,
    lambda_div: float = 0.5
) -> List[int]:
    """
    Maximal Marginal Relevance:
      select k items maximizing lambda * sim(candidate, query)
                              - (1-lambda) * max sim(candidate, selected)
    All sims are cosine (we assume normalized).
    Returns positions into cand_ids.
    """
    if not cand_ids:
        return []
    k = min(k, len(cand_ids))
    q = query_vec.astype("float32")
    sim_to_q = (cand_vecs @ q.T).ravel()  # (n,)

    selected = []
    remaining = list(range(len(cand_ids)))
    first = int(np.argmax(sim_to_q))
    selected.append(first)
    remaining.remove(first)

    cc_sim = cand_vecs @ cand_vecs.T  # (n, n)

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -1e9
        for j in remaining:
            div = np.max(cc_sim[j, selected])
            mmr = lambda_div * sim_to_q[j] - (1.0 - lambda_div) * div
            if mmr > best_score:
                best_score = mmr
                best_idx = j
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

# ---------------------------
# Load indexes, metadata, and embeddings
# ---------------------------
print("Loading FAISS and metadata...")
text_index = load_faiss(TEXT_INDEX_PATH)
img_index  = load_faiss(IMAGE_INDEX_PATH) if IMAGE_INDEX_PATH.exists() else None

text_meta = load_meta_csv(TEXT_META_CSV)           # rows aligned with FAISS ids
img_meta  = load_meta_csv(IMG_META_CSV)
chunk_by_id = load_text_chunks_map(TEXT_CHUNKS_JSONL)

if not TEXT_EMB_NPY.exists():
    raise FileNotFoundError(f"Missing text embeddings matrix: {TEXT_EMB_NPY}")
TEXT_EMB = np.load(TEXT_EMB_NPY).astype("float32")  # already normalized in Phase 4 (we'll rely on that)

print(f"Text vectors: {text_index.ntotal}, Image vectors: {img_index.ntotal if img_index else 0}")
print(f"Text meta rows: {len(text_meta)}, Text chunks in map: {len(chunk_by_id)}")

# ---------------------------
# Retrieval (with MMR, per-doc cap, and optional image hits)
# ---------------------------
def retrieve(query: str,
            k_text=8, k_img=4,
            mmr_pool=40, lambda_div=0.55,
            per_doc_cap=2) -> List[Dict[str, Any]]:

    # 1) Text search, get a larger pool
    q_openai = embed_query_openai(query)  # (1, d)
    t_scores, t_ids = faiss_search(text_index, q_openai, top_k=max(k_text, mmr_pool))
    text_pool = [(int(idx), float(sc)) for idx, sc in zip(t_ids, t_scores)
                 if idx != -1 and idx < len(text_meta)]

    # 2) MMR diversification on text pool
    results: List[Dict[str, Any]] = []
    if text_pool:
        cand_ids = [idx for idx, _ in text_pool]
        cand_vecs = TEXT_EMB[np.array(cand_ids)]
        sel_pos = mmr_select(q_openai, cand_ids, cand_vecs, k=k_text, lambda_div=lambda_div)
        text_sel = [text_pool[p] for p in sel_pos]

        # 3) Per‑document cap
        per_doc_count: Dict[str, int] = {}
        for idx, sc in text_sel:
            row = text_meta.iloc[idx].to_dict()
            fn = row.get("file_name", "")
            if per_doc_count.get(fn, 0) >= per_doc_cap:
                continue
            per_doc_count[fn] = per_doc_count.get(fn, 0) + 1
            row.update({"_score": float(sc), "_modality": "text", "_faiss_id": int(idx)})
            results.append(row)
            if len(results) >= k_text:
                break

    # 4) Optional image search (text→image via CLIP)
    if img_index is not None and img_index.ntotal > 0 and not img_meta.empty and k_img > 0:
        q_clip = clip_text_embed([query])
        i_scores, i_ids = faiss_search(img_index, q_clip, top_k=k_img)
        for idx, sc in zip(i_ids, i_scores):
            if idx == -1 or idx >= len(img_meta):
                continue
            row = img_meta.iloc[int(idx)].to_dict()
            row.update({"_score": float(sc), "_modality": "image", "_faiss_id": int(idx)})
            results.append(row)

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results

# ---------------------------
# Rerank with CrossEncoder (optional)
# ---------------------------
_reranker = None
def rerank_text_hits(query: str, text_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank diversified text hits using a cross-encoder over the FULL chunk text."""
    if not USE_RERANK or not text_hits:
        return text_hits
    global _reranker
    from sentence_transformers import CrossEncoder
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    # Build pairs (query, text) from the original chunk text
    pairs = []
    idx_to_text = []
    for h in text_hits:
        cid = h.get("chunk_id")
        rec = chunk_by_id.get(cid, {})
        txt = rec.get("text", "")
        if not txt:
            txt = f"{h.get('file_name','')} pages {h.get('page_start','?')}-{h.get('page_end','?')} {h.get('heading_path','')}"
        pairs.append([query, txt])
        idx_to_text.append(txt)

    scores = _reranker.predict(pairs, show_progress_bar=False)
    for h, sc in zip(text_hits, scores):
        h["_rerank"] = float(sc)

    text_hits.sort(key=lambda x: x.get("_rerank", 0.0), reverse=True)
    return text_hits

# ---------------------------
# Build context and answer
# ---------------------------
def build_context(results: List[Dict[str, Any]], char_limit: int = 4500) -> Tuple[str, List[Dict[str, Any]]]:
    pieces = []
    kept = []
    for r in results:
        if r.get("_modality") != "text":
            continue
        cid = r.get("chunk_id")
        rec = chunk_by_id.get(cid)
        snippet = rec.get("text", "") if rec else ""
        if not snippet:
            continue
        if sum(len(p) for p in pieces) + len(snippet) > char_limit:
            break
        pieces.append(snippet)
        kept.append(r)
    return ("\n\n---\n\n".join(pieces), kept)

def answer_with_openai(query: str, context: str) -> str:
    system = (
        "You are a helpful assistant that answers strictly from the provided context. "
        "If the context does not contain the answer, say you don't know. "
        "Cite filenames and page ranges when useful."
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=msgs
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python phase5_retrieve_answer.py \"your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\n🔎 Query: {query}")

    # Retrieve with diversification
    hits = retrieve(query, k_text=8, k_img=4, mmr_pool=50, lambda_div=0.55, per_doc_cap=2)

    # Keep only text hits for context building, rerank them, then context
    text_hits = [h for h in hits if h.get("_modality") == "text"]
    text_hits = rerank_text_hits(query, text_hits)  # cross-encoder rerank (optional)

    ctx, used = build_context(text_hits, char_limit=4500)
    if not used:
        print("\nNo text context found. Try a different query.")
        sys.exit(0)

    print(f"\nUsing {len(used)} text chunks for context.")
    ans = answer_with_openai(query, ctx)

    print("\n🧠 Answer:\n" + ans)
    print("\n📚 Sources:")
    for i, h in enumerate(used, 1):
        fn = h.get("file_name", "?")
        ps = f"{h.get('page_start','?')}-{h.get('page_end','?')}"
        hp = h.get("heading_path", "")
        print(f"  {i}. {fn} (pages {ps}) {('• ' + hp) if hp else ''}")
