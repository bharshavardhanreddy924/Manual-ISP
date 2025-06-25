# Image Signal Processing (ISP)

## Overview

This project is an **Image Signal Processing (ISP) Tuning Tool** that implements basic ISP routines for sensor RAW images, with configurable parameters to visualize and compare different processing steps interactively. The tool leverages **OpenCV** and **Streamlit** for image processing and a user-friendly GUI, respectively.

---

## Features

- **Demosaicing**: Edge-based interpolation (5x5) for computing missing channels.
- **White Balance**: Gray World algorithm to remove color casts.
- **Denoising**: Gaussian filter (5x5) for noise reduction with adjustable kernel size and sigma.
- **Gamma Correction**: Uses sRGB gamma correction to convert 12-bit images to 8-bit.
- **Sharpening**: Unsharp mask filter with customizable strength.
- **Interactive UI**: A tuning tool to adjust parameters of the algorithm blocks and visualize results in real time.

---

## Input and Output

- **Input**: 12-bit Bayer RAW image (`.raw` or `.bin`) with:
  - Bayer pattern: **GRBG**
  - Resolution: **1920x1280**
- **Output**: RGB image with **24 bits per pixel** (8 bits for each channel).

---

## To Run

1. Run this command after cloning the repo:
   ```bash
   streamlit run ISP.py

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/dattaaaaa/PixelFusion_Emmetra.git
   cd PixelFusion_Emmetra

---

  # How to Use

1. Launch the tool and upload a 12-bit RAW Bayer image.  
2. Choose between:  
   - **Interactive Tuning**: Adjust parameters like white balance, denoising, gamma correction, and sharpening interactively.  
   - **Predefined Combinations**: Quickly apply predefined processing pipelines, such as:  
     - Demosaic + Gamma  
     - Demosaic + White Balance + Gamma  
     - Demosaic + White Balance + Denoise + Gamma  
     - Demosaic + White Balance + Denoise + Gamma + Sharpening  
3. Visualize the intermediate and final processed results.

---

## Tools for Viewing RAW Files

Use one of the following tools to view your RAW images:  
- PixelViewer  
- IrfanView (with RAW plugin)  
- Configuration to be used for input – Bayer –
12bits, GRBG, 1920x1280

---

## Example Results

### Input  
A 12-bit Bayer RAW image.

### Processing Stages  
- **Demosaic**: Converts the Bayer RAW to RGB.  
- **White Balance**: Adjusts color channels for a neutral appearance.  
- **Denoise**: Smoothens noise using Gaussian filtering.  
- **Gamma Correction**: Enhances brightness and contrast.  
- **Sharpening**: Improves image details.  

### Output  
The final processed image along with intermediate output images at each stage are displayed interactively in the Streamlit application.

#### Output image for Demosaic + White Balance + Denoise + Gamma + Sharpening:

![Output_Final_A1](https://github.com/user-attachments/assets/f5c2c4f3-154e-49d5-bd92-1bffb6115ad9)
