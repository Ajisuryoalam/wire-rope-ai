import os
import streamlit as st
from ingest_pdf import extract_pages_as_images
from build_index import build_index_from_metadata
from search_engine import search_by_text

st.set_page_config(page_title="Wire Rope AI", layout="wide")
st.title("Wire Rope & Fractography AI Assistant")
st.write("Upload PDF referensi wire rope / fractography dan lakukan pencarian gambar.")

# ===== Step 1: Pastikan folder data ada =====
os.makedirs("data/pdf", exist_ok=True)
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)
os.makedirs("data/index", exist_ok=True)

# ===== Step 2: Upload PDF =====
uploaded_pdf = st.file_uploader("Upload PDF referensi", type=["pdf"])

if uploaded_pdf is not None:
    pdf_path = os.path.join("data/pdf", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"File {uploaded_pdf.name} berhasil diupload!")

    # ===== Step 3: Generate FAISS index =====
    INDEX_PATH = "data/index/image_index.faiss"
    st.info("Generating FAISS index from uploaded PDF, please wait...")
    metadata = extract_pages_as_images(pdf_path)
    build_index_from_metadata(metadata)
    st.success("Index generation completed!")

# ===== Step 4: Input query =====
query = st.text_input("Masukkan kata kunci", value="bending fatigue break on wire rope")
top_k = st.slider("Jumlah referensi gambar", 3, 10, 5)

# ===== Step 5: Search button =====
if st.button("Cari referensi"):
    if not os.path.exists("data/index/image_index.faiss"):
        st.warning("Index belum tersedia. Upload PDF dan generate index terlebih dahulu.")
    else:
        results = search_by_text(query, top_k=top_k)
        if len(results) == 0:
            st.warning("Tidak ada hasil ditemukan.")
        else:
            for r in results:
                st.divider()
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(r["image_path"], caption=f"{r['book']} - page {r['page']}")
                with col2:
                    st.write(f"Similarity score: {r['score']:.3f}")
                    st.write(f"Book: {r['book']}")
                    st.write(f"Page: {r['page']}")
                    st.write("Context:")
                    st.write(r["context"])
