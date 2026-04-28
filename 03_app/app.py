import streamlit as st
import datetime
import hashlib
import base64
import os
import textwrap
from PIL import Image

from constants import EN_TO_ES, COUNTRY_REGION_MAP, CATEGORY_MAP, SHIPPING_DAYS_MAP
from logic import load_prediction_model, prepare_input_data

# --- AYARLAR VE YÜKLEMELER ---

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(CURRENT_DIR)

logo_path = os.path.join(CURRENT_DIR, "logo.png")

model_path = os.path.join(BASE_DIR, "04_models", "lightgbm_late_delivery.pkl")

style_path = os.path.join(CURRENT_DIR, "style.css")

def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_base64 = get_base64_image(logo_path)

try:
    favicon = Image.open(logo_path)
except:
    favicon = "🚚"

st.set_page_config(page_title="WhenOA | Smart Supply Chain", page_icon=favicon, layout="wide")

# CSS Uygulama
if os.path.exists(style_path):
    with open(style_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

model = load_prediction_model(model_path)

# --- BAŞLIK  ---
st.markdown(f"""
<div style="display: flex; align-items: flex-start; gap: 20px; margin-bottom: 5px;">
    <img src="data:image/png;base64,{logo_base64}" width="65" style="object-fit: contain; margin-top: 5px;">
    <div style="display: flex; flex-direction: column; justify-content: center;">
        <p class="main-header" style="margin:0; line-height: 1.2;">WhenOA</p>
        <p class="sub-header-tr" style="margin:0; line-height: 1.2;">Kestirimci Tedarik Zinciri Analitiği</p>
        <p class="sub-header-en" style="margin:0; line-height: 1.2;">Predictive Supply Chain Analytics</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# --- GİRİŞ ALANLARI ---
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown('<p class="section-label"><span class="header-icon">🕒</span> Zamanlama | Timing</p>', unsafe_allow_html=True)
    shipping_mode = st.selectbox("Kargo Modu | Shipping Mode", list(SHIPPING_DAYS_MAP.keys()))
    order_date = st.date_input("Sipariş Tarihi | Order Date", value=datetime.date.today())
    order_hour = st.slider("Sipariş Saati | Order Hour", 0, 23, 14)

with c2:
    st.markdown('<p class="section-label"><span class="header-icon">📍</span> KONUM & MÜŞTERİ | INFO</p>', unsafe_allow_html=True)
    order_country_en = st.selectbox("Sipariş Ülkesi | Order Country", sorted(EN_TO_ES.keys()))
    order_country_es = EN_TO_ES[order_country_en]
    order_region = COUNTRY_REGION_MAP[order_country_es]
    st.markdown(f'<span class="region-badge">{order_region}</span>', unsafe_allow_html=True)
    payment_type = st.selectbox("Ödeme | Payment", ['CASH', 'DEBIT', 'PAYMENT', 'TRANSFER'])
    customer_segment = st.selectbox("Segment", ['Consumer', 'Corporate', 'Home Office'])

with c3:
    st.markdown('<p class="section-label"><span class="header-icon">📦</span> Ürün | Product</p>', unsafe_allow_html=True)
    category = st.selectbox("Kategori | Category", sorted(CATEGORY_MAP.keys()))
    sc1, sc2 = st.columns(2)
    with sc1:
        product_price = st.number_input("Fiyat | Price (USD)", min_value=0.0, value=99.99)
    with sc2:
        item_quantity = st.number_input("Miktar | Quantity", min_value=1, max_value=10, value=1)
    discount_rate = st.slider("İndirim | Discount", 0.0, 0.25, 0.0, 0.01)

st.divider()

# --- TAHMİN MANTIĞI ---

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button(
        "🔍 Gecikme Riskini Hesapla | Predict Delivery Risk", 
        use_container_width=True, # Genişliği kutuya uydur (ama CSS sınır koyacak)
        type="primary"
    )

if predict_btn:
    if model is not None:
        input_df = prepare_input_data(
            product_price, item_quantity, SHIPPING_DAYS_MAP, shipping_mode,
            order_hour, order_date, payment_type, customer_segment,
            order_region, order_country_es, CATEGORY_MAP[category],
            discount_rate 
        )

        probability = model.predict_proba(input_df)[0][1]
        threshold = 0.40
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ref_code = hashlib.sha256(f"{order_country_es}-{probability}".encode()).hexdigest()[:12]

        status_class = "res-card-late" if probability >= threshold else "res-card-ontime"
        status_text = "Gecikmeli / Late Delivery" if probability >= threshold else "Zamanında / On Time"
        color_class = "color-late" if probability >= threshold else "color-ontime"

        st.markdown(f"""
        <div class="result-container">
            <div class="res-card {status_class}">
                <p class="res-title {color_class}">{status_text}</p>
                <p class="res-prob {color_class}">%{probability*100:.1f}</p>
                <p class="res-sub">Gecikme olasılığı / Late delivery probability</p>
                <p class="res-ref">Ref: {ref_code} | {timestamp}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model dosyası bulunamadı, tahmin yapılamıyor.")

# --- FOOTER ---
footer_html = textwrap.dedent(f"""
    <div style="text-align: left; border-top: 1px solid #1f2937; margin-top: 4rem; padding-top: 2rem;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{logo_base64}" width="25" style="opacity: 0.8;">
            <span style="font-family: 'Syne', sans-serif; font-weight: 700; color: #ffffff; font-size: 0.8rem;">WhenOA</span>
        </div>
        <div style="font-size: 0.8rem; color: #4b5563; line-height: 1.8; margin-top: 15px;">
            <div style="margin-bottom: 5px;"><strong>Model Referans Veri Seti:</strong> <a href="https://data.mendeley.com/datasets/8gx2fvg2k6/5" target="_blank" class="footer-link">DataCo Supply Chain Dataset</a></div>
            <div style="margin-bottom: 5px;"><strong>Proje Detayları:</strong> <a href="https://github.com/ishilmioz/supply-chain-predictive-analytics" target="_blank" class="footer-link">GitHub</a></div>
            <div style="margin-bottom: 5px;"><strong>İletişim:</strong> <a href="mailto:ismail@hilmiozcelik.com" class="footer-link">ismail@hilmiozcelik.com</a></div>
        </div>
        <p style="font-size: 0.75rem; color: #374151; margin-top: 20px; line-height: 1.4;">
            © 2026 WhenOA | Bu uygulama makine öğrenmesi tabanlı tahmin yapar, kesin sonuç garantisi vermez.<br>
            <span>© 2026 WhenOA | This application provides ML-based predictions and does not guarantee results.</span>
        </p>
    </div>
""")

st.markdown(footer_html, unsafe_allow_html=True)