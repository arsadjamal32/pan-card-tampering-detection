import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
import io
import cv2

# ── Page config ──
st.set_page_config(
    page_title="PAN Card Tampering Detection",
    page_icon="🔍",
    layout="centered"
)

# ── Load model ──
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_pan_model.keras')
    return model

# ── ELA function ──
def get_ela(image_pil, quality=90, scale=15):
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    ela = ImageChops.difference(image_pil, compressed)
    extrema = ela.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff * scale)
    return ela

# ── Preprocess ──
def preprocess(image_pil, target_size=(224, 224)):
    img_rgb = image_pil.convert('RGB').resize(target_size)
    orig_arr = np.array(img_rgb) / 255.0
    ela = get_ela(img_rgb)
    ela_arr = np.array(ela.resize(target_size)) / 255.0
    combined = np.concatenate([orig_arr, ela_arr], axis=-1)
    return np.expand_dims(combined, axis=0)

# ── UI ──
st.title("🔍 PAN Card Tampering Detection")
st.markdown("PAN card ki image upload karo — genuine hai ya tampered, pata chalega.")

st.divider()

uploaded = st.file_uploader(
    "PAN Card image upload karo",
    type=['jpg', 'jpeg', 'png']
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("ELA Analysis")
        ela_img = get_ela(image.convert('RGB'))
        st.image(ela_img, use_column_width=True)
        st.caption("Bright regions = possible tampering")

    st.divider()

    with st.spinner("Analyzing..."):
        try:
            model = load_model()
            input_data = preprocess(image)
            prob = model.predict(input_data, verbose=0)[0][0]
            threshold = 0.30

            st.subheader("Result")

            if prob > threshold:
                confidence = prob * 100
                st.error(f"⚠️ TAMPERED detected!")
                st.metric("Tampering probability", f"{confidence:.1f}%")
                st.progress(float(prob))
                st.warning(
                    "Is PAN card mein tampering ke signs mile hain. "
                    "Manual verification recommended hai."
                )
            else:
                confidence = (1 - prob) * 100
                st.success(f"✅ GENUINE lag raha hai")
                st.metric("Genuine probability", f"{confidence:.1f}%")
                st.progress(float(1 - prob))
                st.info(
                    "Koi obvious tampering nahi mili. "
                    "Phir bhi important documents ke liye "
                    "manual verification karo."
                )

            # Probability bar
            st.divider()
            st.subheader("Detailed Analysis")
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Genuine score",
                          f"{(1-prob)*100:.1f}%")
            with col4:
                st.metric("Tampered score",
                          f"{prob*100:.1f}%")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure 'best_pan_model.keras' same folder mein hai.")

st.divider()
st.caption(
    "Note: Ye tool 71% accuracy ke saath kaam karta hai. "
    "Critical decisions ke liye sirf isi pe depend mat karo."
)