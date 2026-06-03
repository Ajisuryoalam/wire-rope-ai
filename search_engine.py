import json
import faiss
import torch
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

INDEX_PATH = Path("data/index/image_index.faiss")
META_PATH = Path("data/index/image_metadata.json")

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

index = faiss.read_index(str(INDEX_PATH))
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def embed_text(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def search_by_text(query, top_k=5):
    q = embed_text(query)
    scores, ids = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        item = metadata[idx]
        results.append({
            "score": float(score),
            "book": item["book"],
            "page": item["page"],
            "image_path": item["image_path"],
            "context": item["page_text"][:800]
        })
    return results
