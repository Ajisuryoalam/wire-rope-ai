import fitz
import json
from pathlib import Path

OUT_IMG = Path("data/images")
OUT_META = Path("data/metadata")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_META.mkdir(parents=True, exist_ok=True)

def extract_pages_as_images(pdf_path: str, dpi: int = 200):
    doc = fitz.open(pdf_path)
    metadata = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image_name = f"{Path(pdf_path).stem}_page_{page_index+1:04d}.png"
        image_path = OUT_IMG / image_name
        pix.save(image_path)

        text = page.get_text("text")

        metadata.append({
            "book": Path(pdf_path).name,
            "page": page_index + 1,
            "image_path": str(image_path),
            "page_text": text[:3000]
        })

    meta_path = OUT_META / f"{Path(pdf_path).stem}_pages.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata
