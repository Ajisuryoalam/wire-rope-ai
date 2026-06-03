import json
import faiss
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

META_PATH = Path("data/metadata/all_pages.json")
INDEX_PATH = Path("data/index/image_index.faiss")
EMB_PATH = Path("data/index/image_metadata.json")

INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")[0]

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

vectors = []
valid_meta = []

for item in metadata:
    image_path = item["image_path"]
    try:
        vec = embed_image(image_path)
        vectors.append(vec)
        valid_meta.append(item)
    except Exception as e:
        print(f"Skip {image_path}: {e}")

vectors = np.vstack(vectors)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, str(INDEX_PATH))

with open(EMB_PATH, "w", encoding="utf-8") as f:
    json.dump(valid_meta, f, indent=2, ensure_ascii=False)

print(f"Indexed {len(valid_meta)} images.")
