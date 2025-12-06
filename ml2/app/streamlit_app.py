import streamlit as st
import pandas as pd
# di paling atas app/streamlit_app.py
import sys
from pathlib import Path

# tambahkan parent (project root) ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# pastikan project root ada di sys.path sehingga import `ml2` berhasil
sys.path.append(str(PROJECT_ROOT))

# sekarang import
from ml2.utils import recommend_laptops

# gunakan `PROJECT_ROOT` untuk membangun path ke dataset
ROOT = PROJECT_ROOT
df = pd.read_csv(ROOT / "ml2" / "datasets" / "laptops.csv")

st.title("Laptop Recommender — Prototype")
st.write("Masukkan preferensi untuk mendapatkan rekomendasi.")

ram = st.slider("Minimum RAM (GB)", 4, 64, 8, 4)
budget = st.number_input("Budget (USD)", min_value=0, value=1200)
gpu_input = st.text_input("Prefer GPU keywords (comma)", "rtx,gtx")
cpu_input = st.text_input("Prefer CPU keywords (comma)", "i5,ryzen")

if st.button("Recommend"):
    user_need = {
        "min_ram": int(ram),
        "prefer_gpu": [s.strip() for s in gpu_input.split(",") if s.strip()],
        "prefer_cpu": [s.strip() for s in cpu_input.split(",") if s.strip()]
    }
    rec = recommend_laptops(df=df, user_need=user_need, budget=float(budget), top_n=5)
    st.table(rec[["brand","model","pred_price","score"]])
    for i, row in rec.iterrows():
        st.write(f"**{row['brand']} {row['model']}** — Score: {row['score']}")
        st.write(row["reason"])
        st.write("---")
