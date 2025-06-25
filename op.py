import streamlit as st
import numpy as np
import cv2

def main():
    st.title("Display Raw Bayer Image (3456x2160)")

    uploaded_file = st.file_uploader("Upload your raw Bayer image (.raw)", type=["raw"])

    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        raw_data = np.frombuffer(raw_bytes, dtype=np.uint16)

        height, width = 2160, 3456
        try:
            bayer_img = raw_data.reshape((height, width))

            # Normalize to 0-255 uint8 for display
            bayer_8bit = cv2.normalize(bayer_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            st.image(bayer_8bit, caption="Raw Bayer Image (Grayscale)", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to reshape or display image: {e}")

if __name__ == "__main__":
    main()
