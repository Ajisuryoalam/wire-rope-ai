import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def build_index_from_metadata(metadata):
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    vectors, valid_meta = [], []

    for item in metadata:
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            vectors.append(emb.cpu().numpy().astype("float32")[0])
            valid_meta.append(item)
        except:
            continue

    vectors = np.vstack(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs("data/index", exist_ok=True)
    faiss.write_index(index, "data/index/image_index.faiss")
    with open("data/index/image_metadata.json", "w") as f:
        json.dump(valid_meta, f, indent=2, ensure_ascii=False)
