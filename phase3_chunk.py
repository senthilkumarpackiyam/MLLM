# Phase 3 (LangChain) — tokenize-aware chunking via tiktoken + RecursiveCharacterTextSplitter
# pip install -U langchain==0.2.7 langchain-text-splitters==0.2.2 tiktoken

import json
from pathlib import Path
from typing import List, Dict, Any
import re

# ---- Config ----
CHUNK_TOKENS     = 400
CHUNK_OVERLAP    = 60
MIN_CHARS        = 200
MAX_CHARS        = 8000
HEADING_JOIN     = True
PARA_JOIN_SEP    = "\n"
FLUSH_EVERY_PAGES= 8  # safety flush for very long heading-less docs (not critical w/ LC splitters)

# ---- Paths ----
cfg = json.loads(Path("config.json").read_text())
P = {k: Path(v).expanduser().resolve() for k, v in cfg["paths"].items()}
EXTRACT_DIR = P["EXTRACT_DIR"]
CHUNK_DIR   = P["CHUNK_DIR"]; CHUNK_DIR.mkdir(parents=True, exist_ok=True)
TEXT_CHUNKS  = CHUNK_DIR / "text_chunks.jsonl"
IMAGE_CHUNKS = CHUNK_DIR / "image_chunks.jsonl"
SUMMARY_JSON = CHUNK_DIR / "chunk_summary.json"

# ---- LangChain splitters (token-length aware)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")
def tok_len(s: str) -> int:  # ~OpenAI tokenization
    return len(_enc.encode(s or ""))

def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=tok_len,
        separators=["\n\n", "\n", " ", ""],  # coarse → fine
        is_separator_regex=False,
    )

def normalize_heading(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return re.sub(r"[:\-–—]+\s*$", "", t)

def heading_path(stack: List[str]) -> str:
    clean = [h for h in (normalize_heading(x) for x in stack) if h]
    return " > ".join(clean)

def load_pages(doc_dir: Path) -> List[Dict[str, Any]]:
    f = doc_dir / "pages.jsonl"
    if not f.exists(): return []
    out = []
    with open(f, "r", encoding="utf-8", errors="ignore") as r:
        for line in r:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out

def iter_doc_dirs(base: Path) -> List[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir() and (p / "pages.jsonl").exists()])

def main():
    doc_dirs = iter_doc_dirs(EXTRACT_DIR)
    if not doc_dirs:
        print("No extracted docs. Run phase2_extract.py first.")
        return

    splitter = make_splitter()
    text_chunks, image_chunks = [], []
    print(f"Found {len(doc_dirs)} documents to chunk...\n")

    for i, d in enumerate(doc_dirs, 1):
        # meta
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {"id": d.name, "file_name": d.name}
        doc_id = meta.get("id", d.name)
        file_name = meta.get("file_name", d.name)
        print(f"[{i}/{len(doc_dirs)}] Processing: {file_name} (id={doc_id})")

        pages = load_pages(d)
        if not pages:
            continue

        # record images (straight-through)
        for page in pages:
            pno = int(page.get("page", 0))
            heads = [normalize_heading(h.get("text","")) for h in (page.get("headings") or []) if h.get("text")]
            ctx_title = heads[-1] if heads else None
            for im in (page.get("images") or []):
                ip = im.get("path")
                if ip:
                    image_chunks.append({
                        "image_id": f"{doc_id}::p{pno:04d}::img{im.get('xref', -1)}",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page": pno,
                        "image_path": ip,
                        "bboxes": im.get("bboxes", []),
                        "context_title": ctx_title,
                        "context_snippet": None
                    })

        # accumulate text across pages, split with token-aware splitter
        stack: List[str] = []
        buffer_pages: List[int] = []
        buffer_texts: List[str] = []
        chunk_seq = 0

        def flush_buffer():
            nonlocal buffer_pages, buffer_texts, chunk_seq, stack
            if not buffer_texts:
                return
            full = PARA_JOIN_SEP.join(buffer_texts)[:MAX_CHARS]
            if len(full) < MIN_CHARS and not stack:
                buffer_pages, buffer_texts = [], []
                return

            docs = splitter.create_documents([full])
            pmin, pmax = min(buffer_pages), max(buffer_pages)
            hpath = heading_path(stack) if HEADING_JOIN else (stack[-1] if stack else "")
            for di, doc in enumerate(docs):
                text = doc.page_content
                text_chunks.append({
                    "chunk_id": f"{doc_id}::p{pmin:04d}-{pmax:04d}::w{chunk_seq:03d}",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_start": int(pmin),
                    "page_end": int(pmax),
                    "heading_path": hpath,
                    "text": text,
                    "char_len": len(text),
                    "est_tokens": tok_len(text)
                })
                chunk_seq += 1

            buffer_pages, buffer_texts = [], []

        for page in pages:
            pno = int(page.get("page", 0))
            ptext = (page.get("text") or "").strip()
            print(f"   ...processed page {pno} (chars={len(ptext)})")

            # detect top-level heading change → flush
            page_heads = [normalize_heading(h.get("text","")) for h in (page.get("headings") or []) if h.get("text")]
            if page_heads:
                flush_buffer()
                stack = [h for h in page_heads if h][:1]

            if ptext:
                buffer_pages.append(pno)
                buffer_texts.append(ptext)

            # periodic safety flush for very long sequences
            if FLUSH_EVERY_PAGES and (len(buffer_pages) % FLUSH_EVERY_PAGES == 0):
                flush_buffer()
                print(f"   ...forced flush at page {pno}, total text chunks: {len(text_chunks)}")

        flush_buffer()
        print(f"   => Chunks so far: {len(text_chunks)} text, {len(image_chunks)} images\n")

    # ---- Save outputs
    with open(TEXT_CHUNKS, "w", encoding="utf-8") as w:
        for rec in text_chunks:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(IMAGE_CHUNKS, "w", encoding="utf-8") as w:
        for rec in image_chunks:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "docs": len(doc_dirs),
        "text_chunks": len(text_chunks),
        "image_chunks": len(image_chunks),
        "avg_text_chunk_tokens": round(sum(r["est_tokens"] for r in text_chunks)/max(1,len(text_chunks)), 1) if text_chunks else 0,
        "max_text_chunk_tokens": max([r["est_tokens"] for r in text_chunks] or [0]),
        "config": {
            "CHUNK_TOKENS": CHUNK_TOKENS,
            "CHUNK_OVERLAP": CHUNK_OVERLAP,
            "MIN_CHARS": MIN_CHARS,
            "MAX_CHARS": MAX_CHARS
        }
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Chunking Summary (LangChain) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {TEXT_CHUNKS} and {IMAGE_CHUNKS}")

if __name__ == "__main__":
    main()
