import numpy as np
import cv2
import streamlit as st
import tempfile
import os

class ImageSignalProcessor:
    def __init__(self, raw_image):
        self.raw_image = raw_image
        self.height, self.width = raw_image.shape
        
    def demosaic_fast(self):
        """
        Fast Demosaicing using OpenCV.
        Assumes the Bayer pattern is GRBG. Adjust the pattern if necessary.
        """
        # Ensure input is 16-bit integer for OpenCV
        bayer_image = self.raw_image.astype(np.uint16)
        # Perform demosaicing using OpenCV
        rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BayerGR2RGB)
        return rgb_image.astype(np.float32)  # Convert to float32 for consistency
    
    def white_balance_gray_world(self, rgb_image):
        """
        Simple Gray World white balance algorithm
        """
        R, G, B = cv2.split(rgb_image.astype(np.float32))
        
        # Calculate mean of each channel
        R_mean = np.mean(R)
        G_mean = np.mean(G)
        B_mean = np.mean(B)
        
        # Calculate global gain
        K = (R_mean + G_mean + B_mean) / 3
        
        # Apply white balance
        R_balanced = np.clip(R * (K / R_mean), 0, 4095)
        G_balanced = np.clip(G * (K / G_mean), 0, 4095)
        B_balanced = np.clip(B * (K / B_mean), 0, 4095)
        
        return cv2.merge([R_balanced, G_balanced, B_balanced])
    
    def denoise_gaussian(self, rgb_image, kernel_size=5, sigma=1.0):
        """
        Gaussian filter for denoising
        """
        # Convert to 8-bit for processing
        rgb_8bit = (rgb_image / 16).astype(np.uint8)
        denoised_image = cv2.GaussianBlur(rgb_8bit, (kernel_size, kernel_size), sigma)
        
        # Convert back to 12-bit
        return (denoised_image.astype(np.float32) * 16).astype(np.float32)
    
    def normalize_for_display(self, image):
        """
        Normalize image for display between 0 and 1
        """
        # For 12-bit images
        min_val = np.min(image)
        max_val = np.max(image)
        
        if min_val == max_val:
            return np.zeros_like(image)
        
        normalized = (image - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    def gamma_correction(self, rgb_image, gamma=2.2):
        """
        sRGB Gamma correction (12-bit to 8-bit conversion)
        """
        # Normalize to 0-1 range
        normalized = rgb_image / 4095.0  # 12-bit to float
        
        # Apply gamma correction
        corrected = np.power(normalized, 1/gamma)
        
        # Scale back to 0-255
        return np.clip(corrected * 255, 0, 255).astype(np.uint8)
    
    def sharpen_unsharp_mask(self, rgb_image, amount=1.5, radius=1):
        """
        Unsharp mask filter for sharpening
        """
        # Convert to 8-bit for processing
        rgb_8bit = (rgb_image / 16).astype(np.uint8)
        
        # Apply sharpening
        blurred = cv2.GaussianBlur(rgb_8bit, (0, 0), radius)
        sharpened = cv2.addWeighted(rgb_8bit, 1 + amount, blurred, -amount, 0)
        
        # Convert back to 12-bit
        return (sharpened.astype(np.float32) * 16).astype(np.float32)

def load_raw_image(file_path, width=1920, height=1280):
    """
    Load 12-bit RAW Bayer image
    """
    raw_data = np.fromfile(file_path, dtype=np.uint16)
    raw_image = raw_data.reshape((height, width))
    return raw_image

def main():
    st.title("Image Signal Processing (ISP) Tuning Tool")
    
    # Predefined processing combinations
    combinations = [
        "Demosaic + Gamma",
        "Demosaic + White Balance + Gamma",
        "Demosaic + White Balance + Denoise + Gamma",
        "Demosaic + White Balance + Denoise + Gamma + Sharpen"
    ]
    
    # File uploader for RAW image
    uploaded_file = st.file_uploader("Choose a 12-bit RAW image", type=['raw', 'bin'])
    
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            # Load RAW image
            raw_image = load_raw_image(tmp_file_path)
            
            # Create ISP processor
            isp = ImageSignalProcessor(raw_image)
            
            # Processing mode selection
            processing_mode = st.radio(
                "Select Processing Mode", 
                ["Interactive Tuning", "Predefined Combinations"]
            )
            
            if processing_mode == "Interactive Tuning":
                # Sidebar for algorithm parameters
                st.sidebar.header("ISP Algorithm Parameters")
                
                # White Balance parameters
                apply_wb = st.sidebar.checkbox("Apply White Balance", value=True)
                
                # Denoise parameters
                apply_denoise = st.sidebar.checkbox("Apply Denoise", value=True)
                denoise_kernel = st.sidebar.slider("Denoise Kernel Size", 3, 7, 5, step=2)
                denoise_sigma = st.sidebar.slider("Denoise Sigma", 0.1, 2.0, 1.0, step=0.1)
                
                # Gamma correction parameters
                gamma_value = st.sidebar.slider("Gamma Value", 1.0, 3.0, 2.2, step=0.1)
                
                # Sharpening parameters
                apply_sharpen = st.sidebar.checkbox("Apply Sharpening", value=False)
                sharpen_amount = st.sidebar.slider("Sharpening Amount", 0.0, 3.0, 1.5, step=0.1)
                
                # Processing pipeline
                st.subheader("Processing Pipeline")
                
                # Demosaic
                rgb_image = isp.demosaic_fast()
                st.image(isp.normalize_for_display(rgb_image), caption="After Demosaic", channels="RGB")
                
                # White Balance
                if apply_wb:
                    rgb_image = isp.white_balance_gray_world(rgb_image)
                    st.image(isp.normalize_for_display(rgb_image), caption="After White Balance", channels="RGB")
                
                # Denoise
                if apply_denoise:
                    rgb_image = isp.denoise_gaussian(rgb_image, kernel_size=denoise_kernel, sigma=denoise_sigma)
                    st.image(isp.normalize_for_display(rgb_image), caption="After Denoise", channels="RGB")
                
                # Gamma Correction
                gamma_corrected_image = isp.gamma_correction(rgb_image, gamma=gamma_value)
                st.image(gamma_corrected_image / 255.0, caption="After Gamma Correction", channels="RGB")
                
                # Sharpening
                if apply_sharpen:
                    rgb_image = isp.sharpen_unsharp_mask(
                        (gamma_corrected_image.astype(np.float32) * 16)
                    )
                    st.image(isp.normalize_for_display(rgb_image), caption="After Sharpening", channels="RGB")
            
            else:  # Predefined Combinations
                # Let user select a combination
                selected_combination = st.selectbox(
                    "Choose a Processing Combination", 
                    combinations
                )
                
                # Perform processing based on selected combination
                st.subheader(f"Processing: {selected_combination}")
                
                # Original RAW image display
                st.image(
                    isp.normalize_for_display(raw_image.astype(np.float32)), 
                    caption="Original RAW Image", 
                    channels="GRAY"
                )
                
                # Demosaic
                rgb_image = isp.demosaic_fast()
                st.image(
                    isp.normalize_for_display(rgb_image), 
                    caption="After Demosaic", 
                    channels="RGB"
                )
                
                # White Balance
                if "White Balance" in selected_combination:
                    rgb_image = isp.white_balance_gray_world(rgb_image)
                    st.image(
                        isp.normalize_for_display(rgb_image), 
                        caption="After White Balance", 
                        channels="RGB"
                    )
                
                # Denoise
                if "Denoise" in selected_combination:
                    rgb_image = isp.denoise_gaussian(rgb_image)
                    st.image(
                        isp.normalize_for_display(rgb_image), 
                        caption="After Denoise", 
                        channels="RGB"
                    )
                
                # Gamma Correction
                gamma_corrected_image = isp.gamma_correction(rgb_image)
                st.image(
                    gamma_corrected_image / 255.0, 
                    caption="After Gamma Correction", 
                    channels="RGB"
                )
                
                # Sharpening
                if "Sharpen" in selected_combination:
                    rgb_image = isp.sharpen_unsharp_mask(
                        (gamma_corrected_image.astype(np.float32) * 16)
                    )
                    st.image(
                        isp.normalize_for_display(rgb_image), 
                        caption="After Sharpening", 
                        channels="RGB"
                    )
        
        finally:
            # Clean up temporary file
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
