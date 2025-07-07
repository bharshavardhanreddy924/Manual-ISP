import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

class ImageProcessor:
    @staticmethod
    def apply_gray_world(image):
        b, g, r = cv2.split(image)
        b_gain = np.mean(g) / np.mean(b) if np.mean(b) > 0 else 1
        g_gain = np.mean(g) / np.mean(g) if np.mean(g) > 0 else 1
        r_gain = np.mean(g) / np.mean(r) if np.mean(r) > 0 else 1
        return cv2.merge((
            cv2.multiply(b, b_gain).clip(0,255).astype(np.uint8),
            g,
            cv2.multiply(r, r_gain).clip(0,255).astype(np.uint8),
        ))

    @staticmethod
    def apply_srgb_gamma(image, gamma=2.2):
        img = image.astype(np.float32) / 255.0
        corrected = np.where(
            img <= 0.0031308,
            12.92 * img,
            1.055 * np.power(img, 1.0 / gamma) - 0.055
        )
        return (corrected * 255).clip(0,255).astype(np.uint8)

    @staticmethod
    def apply_unsharp_mask(image, blur_radius, strength):
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

def main():
    st.set_page_config(
        page_title="Image Signal Processing",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark theme CSS
    st.markdown("""
    <style>
    .main { background-color: #2E2E2E; color: white; }
    .stButton button { background-color: #4CAF50; color: white; font-weight: bold; width: 100%; }
    .stButton button:hover { background-color: #45a049; }
    h1, h2, h3 { color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Image Signal Processing")

    # Initialize session state
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "Demosaic"

    # Sidebar
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Raw Image", type=["raw"])
        if uploaded_file and st.button("Load Image"):
            with st.spinner("Loading and processing..."):
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(uploaded_file.getvalue())
                tmp.close()
                try:
                    bayer = np.fromfile(tmp.name, dtype=np.uint16).reshape((1280, 1920))
                    bayer = (bayer >> 4)
                    bayer8 = cv2.normalize(bayer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    rgb = cv2.cvtColor(bayer8, cv2.COLOR_BAYER_GRBG2BGR_EA)
                    st.session_state.original_image = rgb
                    st.session_state.processed_image = rgb.copy()
                    st.session_state.current_stage = "Demosaic"
                    st.success("Image loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp.name)

        st.header("Parameters")
        gamma = st.slider("Gamma", 0.001, 3.0, 1.16, 0.01)
        blur_radius = st.slider("Blur Radius", 1, 9, 5, 2)
        sharpen_strength = st.slider("Sharpen Strength", 0.5, 5.0, 3.8, 0.1)

        if st.session_state.processed_image is not None:
            if st.button("Save Processed Image"):
                ok, buf = cv2.imencode(".png", st.session_state.processed_image)
                if ok:
                    st.download_button(
                        label="Download Image",
                        data=buf.tobytes(),
                        file_name="processed_image.png",
                        mime="image/png",
                    )

    # Processing buttons
    cols = st.columns(6)
    stages = ["Original", "Demosaic", "White Balance", "Denoise", "Gamma Correct", "Sharpen"]
    for col, stage in zip(cols, stages):
        if col.button(stage):
            img = st.session_state.original_image.copy() if stage in ["Original", "Demosaic"] else st.session_state.processed_image
            if stage == "White Balance":
                img = ImageProcessor.apply_gray_world(img)
            elif stage == "Denoise":
                img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
            elif stage == "Gamma Correct":
                img = ImageProcessor.apply_srgb_gamma(img, gamma)
            elif stage == "Sharpen":
                img = ImageProcessor.apply_unsharp_mask(img, blur_radius, sharpen_strength)
            st.session_state.processed_image = img
            st.session_state.current_stage = stage

    # Display current stage
    st.subheader(f"Current Stage: {st.session_state.current_stage}")

    # Display image at 70% size using use_container_width=False and width parameter
    if st.session_state.processed_image is not None:
        img_bgr = st.session_state.processed_image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (int(w * 0.7), int(h * 0.7)))
        st.image(
            Image.fromarray(resized),
            caption="Processed Image",
            width=int(w * 0.7),
            use_container_width=False
        )
    else:
        st.info("Please load an image to begin.")

    # About
    with st.expander("About this app"):
        st.write("""
        **Pipeline**  
        1. Demosaic  
        2. White Balance (Gray World)  
        3. Denoise (Gaussian Blur)  
        4. Gamma Correction (sRGB)  
        5. Sharpen (Unsharp Mask)
        
        Adjust sliders in the sidebar to tweak parameters.
        """)

if __name__ == "__main__":
    main()
