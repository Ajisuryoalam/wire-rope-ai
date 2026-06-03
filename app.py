import streamlit as st
from search_engine import search_by_text

st.set_page_config(page_title="Wire Rope AI", layout="wide")
st.title("Wire Rope & Fractography AI Assistant")
st.write("Upload foto wire rope atau ketik kata kunci untuk mencari referensi buku.")

query = st.text_input("Masukkan kata kunci", value="bending fatigue break on wire rope")
top_k = st.slider("Jumlah referensi gambar", 3, 10, 5)

if st.button("Cari referensi"):
    results = search_by_text(query, top_k=top_k)

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
