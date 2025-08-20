# Phase 2 — Extract text, headings, images per page
import json, re, sys
from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF

# --- Load config & manifest ---
cfg = json.loads(Path("config.json").read_text())
P = {k: Path(v).expanduser().resolve() for k, v in cfg["paths"].items()}
manifest = pd.read_csv(P["MANIFEST_CSV"])
EXTRACT_DIR = P["EXTRACT_DIR"]; EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(t: str) -> str:
    t = t.replace("\r", " ").replace("\t", " ")
    t = re.sub(r"[ \u00A0]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def guess_headings_from_spans(spans):
    if not spans: return []
    sizes = sorted(float(s.get("size", 10.0)) for s in spans)
    median = sizes[len(sizes)//2] if sizes else 10.0
    p90 = sizes[int(0.9*(len(sizes)-1))] if len(sizes) > 10 else (sizes[-1] if sizes else 10.0)
    out = []
    for s in spans:
        txt = (s.get("text") or "").strip()
        if not txt: 
            continue
        size = float(s.get("size", median))
        is_heading = (len(txt) <= 80) and (size >= max(median + 1.5, 0.9*p90))
        out.append({"text": txt, "bbox": list(s.get("bbox", [])), "size": size, "is_heading": bool(is_heading)})
    return out

def page_to_blocks(page):
    blocks = []
    d = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        spans = []
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                txt = clean_text(sp.get("text", ""))
                if not txt: 
                    continue
                spans.append({
                    "text": txt,
                    "size": float(sp.get("size", 10.0)),
                    "font": sp.get("font", ""),
                    "bbox": list(sp.get("bbox", []))
                })
        if spans:
            para = clean_text(" ".join(s["text"] for s in spans))
            if para:
                blocks.append({"text": para, "spans": spans})
    return blocks

def export_images(page, out_dir: Path, doc_id: str, pno: int):
    out = []
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(page.get_images(full=True) or []):
        xref = img[0]
        try:
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            name = f"{doc_id}_p{pno:04d}_img{i:02d}.png"
            fpath = img_dir / name
            pix.save(fpath)
            out.append({"page": pno, "xref": int(xref), "path": str(fpath), "bboxes": []})
        except Exception as e:
            out.append({"page": pno, "xref": int(xref), "error": str(e)})
    return out

def extract_one(pdf_path: Path, doc_id: str):
    out_dir = EXTRACT_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_jsonl = out_dir / "pages.jsonl"

    meta = {
        "id": doc_id,
        "file_name": pdf_path.name,
        "abs_path": str(pdf_path),
        "out_dir": str(out_dir),
        "page_count": 0,
        "images_count": 0,
        "text_bytes": 0
    }

    images_total = 0
    text_bytes = 0

    with fitz.open(pdf_path) as doc, open(pages_jsonl, "w", encoding="utf-8") as w:
        meta["page_count"] = doc.page_count
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            blocks = page_to_blocks(page)
            spans = [s for b in blocks for s in b["spans"]]
            headings = [h for h in guess_headings_from_spans(spans) if h.get("is_heading")]
            text = clean_text("\n".join(b["text"] for b in blocks if b["text"]))
            text_bytes += len(text.encode("utf-8"))
            imgs = export_images(page, out_dir, doc_id, pno)
            images_total += len([m for m in imgs if "path" in m])

            record = {
                "doc_id": doc_id,
                "file_name": pdf_path.name,
                "page": pno,
                "text": text,
                "blocks": blocks,
                "headings": [{"text": h["text"], "bbox": h["bbox"], "size": h["size"]} for h in headings],
                "images": imgs
            }
            w.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta["images_count"] = images_total
    meta["text_bytes"] = text_bytes
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def main():
    if manifest.empty:
        print("No PDFs found in manifest.csv. Add PDFs to the PDF/ folder and re-run.")
        sys.exit(0)

    corpus_meta = []
    errors = []
    for _, row in manifest.iterrows():
        pdf_path = Path(row["abs_path"])
        doc_id = str(row["id"])
        try:
            meta = extract_one(pdf_path, doc_id)
            corpus_meta.append(meta)
            print(f"OK: {pdf_path.name}  pages={meta['page_count']}  imgs={meta['images_count']}")
        except Exception as e:
            print(f"[ERROR] {pdf_path}: {e}")
            errors.append({"id": doc_id, "file": str(pdf_path), "error": str(e)})

    # Write overall corpus files
    with open(EXTRACT_DIR / "corpus.jsonl", "w", encoding="utf-8") as w:
        for m in corpus_meta:
            w.write(json.dumps(m, ensure_ascii=False) + "\n")

    summary = {
        "docs_processed": len(corpus_meta),
        "docs_failed": len(errors),
        "total_pages": int(sum(m["page_count"] for m in corpus_meta)),
        "total_images": int(sum(m["images_count"] for m in corpus_meta)),
        "total_text_mb": round(sum(m["text_bytes"] for m in corpus_meta) / (1024*1024), 2),
    }
    (EXTRACT_DIR / "summary.json").write_text(
        json.dumps({"summary": summary, "errors": errors}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n=== Extraction Summary ===")
    print(json.dumps(summary, indent=2))
    if errors:
        print("Some files failed; see extracted/summary.json.")

if __name__ == "__main__":
    main()