import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tempfile
import os

class ImageProcessor:
    @staticmethod
    def apply_gray_world(image):
        # Split channels
        b, g, r = cv2.split(image)
        # Compute gains (avoid division by zero)
        b_gain = np.mean(g) / np.mean(b) if np.mean(b) > 0 else 1
        g_gain = np.mean(g) / np.mean(g) if np.mean(g) > 0 else 1
        r_gain = np.mean(g) / np.mean(r) if np.mean(r) > 0 else 1
        # Apply gains and merge back
        return cv2.merge((
            cv2.multiply(b, b_gain).clip(0,255).astype(np.uint8),
            g,
            cv2.multiply(r, r_gain).clip(0,255).astype(np.uint8),
        ))

    @staticmethod
    def apply_srgb_gamma(image, gamma=2.2):
        # Normalize to [0,1]
        img = image.astype(np.float32) / 255.0
        # sRGB transfer
        corrected = np.where(
            img <= 0.0031308,
            12.92 * img,
            1.055 * np.power(img, 1.0 / gamma) - 0.055
        )
        return (corrected * 255).clip(0,255).astype(np.uint8)

    @staticmethod
    def apply_unsharp_mask(image, blur_radius, strength):
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        # Weighted add to sharpen
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

def main():
    st.set_page_config(
        page_title="Image Signal Processing",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main { background-color: #2E2E2E; color: white; }
    .stButton button { background-color: #4CAF50; color: white; font-weight: bold; width: 100%; }
    .stButton button:hover { background-color: #45a049; }
    h1, h2, h3 { color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Image Signal Processing")

    # Session state init
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "Demosaic"

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Raw Image", type=["raw"])
        if uploaded_file is not None and st.button("Load Image"):
            with st.spinner("Loading and processing image..."):
                # Write to temp file
                temp = tempfile.NamedTemporaryFile(delete=False)
                temp.write(uploaded_file.getvalue())
                temp.close()
                try:
                    # Load raw Bayer (1280x1920, 16-bit)
                    bayer = np.fromfile(temp.name, dtype=np.uint16).reshape((1280, 1920))
                    bayer = (bayer >> 4)  # to 12-bit
                    # Normalize to 8-bit
                    bayer8 = cv2.normalize(bayer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    # Demosaic
                    rgb = cv2.cvtColor(bayer8, cv2.COLOR_BAYER_GRBG2BGR_EA)
                    st.session_state.original_image = rgb
                    st.session_state.processed_image = rgb.copy()
                    st.session_state.current_stage = "Demosaic"
                    st.success("Image loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                finally:
                    os.unlink(temp.name)

        st.header("Parameters")
        gamma = st.slider("Gamma", 0.001, 3.0, 1.16, 0.01)
        blur_radius = st.slider("Blur Radius", 1, 9, 5, 2)
        sharpen_strength = st.slider("Sharpen Strength", 0.5, 5.0, 3.8, 0.1)

        if st.session_state.processed_image is not None:
            if st.button("Save Processed Image"):
                try:
                    ok, buf = cv2.imencode(".png", st.session_state.processed_image)
                    if ok:
                        st.download_button(
                            label="Download Image",
                            data=buf.tobytes(),
                            file_name="processed_image.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"Error saving image: {e}")

    # Processing stage buttons
    cols = st.columns(6)
    stages = ["Original", "Demosaic", "White Balance", "Denoise", "Gamma Correct", "Sharpen"]
    for col, stage in zip(cols, stages):
        if col.button(stage):
            if stage == "Original" or stage == "Demosaic":
                st.session_state.processed_image = st.session_state.original_image.copy()
            elif stage == "White Balance":
                st.session_state.processed_image = ImageProcessor.apply_gray_world(
                    st.session_state.processed_image
                )
            elif stage == "Denoise":
                st.session_state.processed_image = cv2.GaussianBlur(
                    st.session_state.processed_image,
                    (blur_radius, blur_radius), 0
                )
            elif stage == "Gamma Correct":
                st.session_state.processed_image = ImageProcessor.apply_srgb_gamma(
                    st.session_state.processed_image, gamma=gamma
                )
            elif stage == "Sharpen":
                st.session_state.processed_image = ImageProcessor.apply_unsharp_mask(
                    st.session_state.processed_image,
                    blur_radius=blur_radius,
                    strength=sharpen_strength
                )
            st.session_state.current_stage = stage

    # Display current stage
    st.subheader(f"Current Stage: {st.session_state.current_stage}")

    # Display processed image at 70% size
    if st.session_state.processed_image is not None:
        # Convert BGR→RGB
        img = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        resized = cv2.resize(img, (int(w * 0.7), int(h * 0.7)))
        # Convert to PIL for better display
        st.image(Image.fromarray(resized), caption="Processed Image", use_column_width=False)
    else:
        st.info("Please load an image to begin processing.")

    # About
    with st.expander("About this app"):
        st.write("""
        This Image Signal Processing app allows you to process raw Bayer images through a pipeline of operations:
        - **Demosaic**: Converts the raw Bayer pattern to a full RGB image
        - **White Balance**: Applies gray world white balancing algorithm
        - **Denoise**: Reduces noise using Gaussian blur
        - **Gamma Correct**: Applies sRGB gamma correction
        - **Sharpen**: Enhances details using unsharp masking
        
        Adjust the sliders in the sidebar to control the processing parameters.
        """)

if __name__ == "__main__":
    main()
