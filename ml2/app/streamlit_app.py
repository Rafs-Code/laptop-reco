import streamlit as st
import pandas as pd
# di paling atas app/streamlit_app.py
import sys
from pathlib import Path

# tambahkan parent (project root) ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml2.utils import recommend_laptops

ROOT = PROJECT_ROOT
df = pd.read_csv(ROOT / "ml2" / "datasets" / "laptops.csv")

# ============================
# CUSTOM UI / THEME
# ============================
st.set_page_config(
    page_title="Laptop Recommender",
    page_icon="💻",
    layout="centered"
)

st.markdown("""
<style>
/* Card Container */
.reco-card {
    padding: 18px 20px;
    border-radius: 12px;
    background: #121212;
    border: 1px solid #333;
    margin-bottom: 15px;
}

/* Title Gradient */
.title-grad {
    font-size: 34px;
    font-weight: 700;
    background: linear-gradient(90deg, #4ca1ff, #c86bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Input Container */
.blockbox {
    padding: 18px;
    background: #181818;
    border-radius: 12px;
    border: 1px solid #333;
}

/* HR clean */
hr {border: none; border-top: 1px solid #333; margin: 15px 0;}
</style>
""", unsafe_allow_html=True)
# ============================
# TITLE
# ============================
st.markdown("<div class='title-grad'>Laptop Recommender</div>", unsafe_allow_html=True)
st.write("Atur preferensi kamu di bawah. Sistem akan mencari laptop paling cocok berdasarkan budget dan kebutuhan.")

# ---------------------------
# SHOW LOGO (gunakan assets/logo-laptop-reco.png)
# ---------------------------
from PIL import Image

# path relatif ke project root (PROJECT_ROOT sudah didefinisikan)
logo_path = PROJECT_ROOT / "assets" / "logo-laptop-reco.png"

if logo_path.exists():
    try:
        logo_img = Image.open(logo_path)
        # header: logo di tengah + judul kecil di samping (2 kolom)
        col_a, col_b, col_c = st.columns([1,4,1])
        with col_a:
            st.write("")  # spacer kiri
        with col_b:
            st.image(logo_img, width=400)
        with col_c:
            st.write("")  # spacer kanan
    except Exception as e:
        st.warning(f"Logo gagal dimuat: {e}")
else:
    # fallback: tidak perlu menampilkan error yang besar, cukup hint kecil
    st.info("Tip: letakkan logo di `assets/logo-laptop-reco.png` untuk menampilkan header khusus.")


# ============================
# INPUT SECTION
# ============================
with st.container():
    st.markdown("<div class='blockbox'>", unsafe_allow_html=True)

    ram = st.slider("Minimum RAM (GB)", 4, 64, 8, 4)
    budget = st.number_input("Budget (USD)", min_value=0, value=1200)
    gpu_input = st.text_input("Prefer GPU keywords (comma)", "rtx, gtx")
    cpu_input = st.text_input("Prefer CPU keywords (comma)", "i5, ryzen")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# RECOMMENDATION OUTPUT
# ============================
if st.button("Recommend", use_container_width=True):
    user_need = {
        "min_ram": int(ram),
        "prefer_gpu": [s.strip() for s in gpu_input.split(",") if s.strip()],
        "prefer_cpu": [s.strip() for s in cpu_input.split(",") if s.strip()]
    }

    rec = recommend_laptops(df=df, user_need=user_need, budget=float(budget), top_n=5)

    st.subheader("Rekomendasi Terbaik")

    for i, row in rec.iterrows():
        st.markdown(f"""
        <div class="reco-card">
            <h3>{row['brand']} {row['model']}</h3>
            <p><b>Score:</b> {row['score']}</p>
            <p><b>Predicted Price:</b> ${row['pred_price']}</p>
            <hr>
            <p>{row['reason']}</p>
        </div>
        """, unsafe_allow_html=True)
