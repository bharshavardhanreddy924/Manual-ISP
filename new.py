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
        b, g, r = cv2.split(image)
        b_gain, g_gain, r_gain = (np.mean(g) / np.mean(c) if np.mean(c) > 0 else 1 for c in (b, g, r))
        return cv2.merge((cv2.multiply(b, b_gain), g, cv2.multiply(r, r_gain)))

    @staticmethod
    def apply_srgb_gamma(image, gamma=2.2):
        image = image / 255.0
        corrected = np.where(image <= 0.0031308, 12.92 * image, 1.055 * (image ** (1 / gamma)) - 0.055)
        return (corrected * 255).clip(0, 255).astype(np.uint8)

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

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #2E2E2E;
        color: white;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Image Signal Processing")

    # Initialize session state for images and processing stage
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "Demosaic"

    # Sidebar for image upload and processing parameters
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Raw Image", type=["raw"])
        
        if uploaded_file is not None:
            if st.button("Load Image"):
                with st.spinner("Loading and processing image..."):
                    # Save the uploaded file to a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.getvalue())
                    temp_file.close()
                    
                    try:
                        # Load and process the Bayer image
                        bayer_image = np.fromfile(temp_file.name, dtype=np.uint16).reshape((1280, 1920))
                        bayer_image = (bayer_image >> 4)  # Shift to 12-bit
                        bayer_image = cv2.normalize(
                            bayer_image, None, 0, 255, cv2.NORM_MINMAX
                        ).astype(np.uint8)
                        
                        # Demosaic the Bayer image to RGB
                        st.session_state.original_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GRBG2BGR_EA)
                        st.session_state.processed_image = st.session_state.original_image.copy()
                        st.session_state.current_stage = "Demosaic"
                        
                        # Clean up the temporary file
                        os.unlink(temp_file.name)
                        
                        st.success("Image loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                        
        st.header("Parameters")
        gamma = st.slider("Gamma", 0.001, 3.0, 1.16, 0.01)
        blur_radius = st.slider("Blur Radius", 1, 9, 5, 2)
        sharpen_strength = st.slider("Sharpen Strength", 0.5, 5.0, 3.8, 0.1)
        
        if st.session_state.processed_image is not None:
            if st.button("Save Processed Image"):
                try:
                    # Convert image to bytes for download
                    is_success, buffer = cv2.imencode(".png", st.session_state.processed_image)
                    if is_success:
                        btn = st.download_button(
                            label="Download Image",
                            data=buffer.tobytes(),
                            file_name="processed_image.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"Error saving image: {e}")

    # Main content area
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("Original"):
            st.session_state.current_stage = "Original"
            st.session_state.processed_image = st.session_state.original_image.copy()
    with col2:
        if st.button("Demosaic"):
            st.session_state.current_stage = "Demosaic"
            st.session_state.processed_image = st.session_state.original_image.copy()
    with col3:
        if st.button("White Balance"):
            if st.session_state.processed_image is not None:
                st.session_state.current_stage = "White Balance"
                st.session_state.processed_image = ImageProcessor.apply_gray_world(st.session_state.processed_image)
    with col4:
        if st.button("Denoise"):
            if st.session_state.processed_image is not None:
                st.session_state.current_stage = "Denoise"
                st.session_state.processed_image = cv2.GaussianBlur(
                    st.session_state.processed_image, (blur_radius, blur_radius), 0
                )
    with col5:
        if st.button("Gamma Correct"):
            if st.session_state.processed_image is not None:
                st.session_state.current_stage = "Gamma Correct"
                st.session_state.processed_image = ImageProcessor.apply_srgb_gamma(
                    st.session_state.processed_image, gamma=gamma
                )
    with col6:
        if st.button("Sharpen"):
            if st.session_state.processed_image is not None:
                st.session_state.current_stage = "Sharpen"
                st.session_state.processed_image = ImageProcessor.apply_unsharp_mask(
                    st.session_state.processed_image, blur_radius=blur_radius, strength=sharpen_strength
                )

    # Display the current processing stage
    st.subheader(f"Current Stage: {st.session_state.current_stage}")

    # Display the image with reduced size (70% of container width)
    if st.session_state.processed_image is not None:
        # Convert the image from BGR to RGB for display
        image_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
        
        # Create a centered column to display the image at 70% width
        col1, col2, col3 = st.columns([0.15, 0.7, 0.15])  # 15% + 70% + 15% = 100%
        with col2:
            st.image(image_rgb, use_container_width=True)
    else:
        st.info("Please load an image to begin processing.")

    # Information about the app
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
