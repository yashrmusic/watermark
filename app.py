import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import easyocr
import base64

# Initialize EasyOCR reader (run once)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA

# Streamlit config
st.set_page_config(page_title="Gemini Watermark Remover", layout="wide")

# Custom CSS with Tailwind (via CDN)
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .header { background: linear-gradient(to right, #4f46e5, #7c3aed); }
        .sidebar { background-color: #f9fafb; }
        .btn-primary { background-color: #4f46e5; color: white; }
        .btn-primary:hover { background-color: #4338ca; }
        .dark-mode { background-color: #1f2937; color: white; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header p-6 text-white text-center">
        <h1 class="text-3xl font-bold">Gemini Watermark Remover</h1>
        <p class="text-sm mt-2">Remove 'ai' watermarks from Gemini-generated images. For educational use only.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar (for ads and settings)
with st.sidebar:
    st.markdown('<div class="sidebar p-4">', unsafe_allow_html=True)
    st.header("Settings & Support")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    debug_mode = st.checkbox("Show debug mask (red rectangle)")
    st.markdown("""
        <h3 class="text-lg font-semibold">Support Us</h3>
        <p class="text-sm">[Your AdSense Ad Here]</p>
        <p class="text-xs text-gray-500">Disclaimer: Watermark removal may violate terms of service. Use responsibly.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Apply dark mode if selected
if theme == "Dark":
    st.markdown('<style>body { background-color: #1f2937; color: white; }</style>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Drag and drop or browse (PNG/JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    # Manual watermark selection
    st.subheader("Manual Watermark Selection")
    use_manual = st.checkbox("Manually select watermark area")
    if use_manual:
        x_start = st.slider("X Start", 0, 1000, 0)
        y_start = st.slider("Y Start", 0, 1000, 0)
        width = st.slider("Width", 10, 200, 50)
        height = st.slider("Height", 10, 200, 50)

if uploaded_file:
    with st.spinner("Processing image..."):
        # Load and display original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess image for OCR (grayscale, contrast enhancement)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        else:
            gray = img_array
        
        # Initialize mask
        mask = np.zeros((img_height, img_width), np.uint8)
        
        if use_manual:
            # Use user-defined region
            cv2.rectangle(mask, (x_start, y_start), (x_start + width, y_start + height), 255, -1)
        else:
            # Auto-detect watermark with EasyOCR
            reader = load_ocr_reader()
            results = reader.readtext(gray, detail=1, paragraph=False)
            watermark_detected = False
            for (bbox, text, conf) in results:
                if 'ai' in text.lower() and conf > 0.3:  # Lowered threshold
                    x, y = int(bbox[0][0]), int(bbox[0][1])
                    w, h = int(bbox[1][0] - bbox[0][0]), int(bbox[1][1] - bbox[0][1])
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    watermark_detected = True
                    break
            
            # Fallback to larger bottom-right region if no watermark detected
            if not watermark_detected:
                st.warning("No 'ai' watermark detected. Using default bottom-right region (50x50px).")
                x_start, y_start = img_width - 50, img_height - 50
                cv2.rectangle(mask, (x_start, y_start), (img_width, img_height), 255, -1)
        
        # Debug mode: Show masked region
        if debug_mode:
            debug_img = img_array.copy()
            if len(debug_img.shape) == 3:
                debug_img[mask == 255] = [255, 0, 0]  # Red mask for color images
            else:
                debug_img[mask == 255] = 255  # White mask for grayscale
            debug_pil = Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB) if len(debug_img.shape) == 3 else debug_img)
            with col2:
                st.image(debug_pil, caption="Debug: Masked Region (Red)", use_column_width=True)
        
        # Inpainting
        try:
            if len(img_array.shape) == 3:  # Color image
                b, g, r = cv2.split(img_array)
                b_inp = cv2.inpaint(b, mask, 3, cv2.INPAINT_TELEA)
                g_inp = cv2.inpaint(g, mask, 3, cv2.INPAINT_TELEA)
                r_inp = cv2.inpaint(r, mask, 3, cv2.INPAINT_TELEA)
                inpainted = cv2.merge([b_inp, g_inp, r_inp])
            else:  # Grayscale
                inpainted = cv2.inpaint(img_array, mask, 3, cv2.INPAINT_TELEA)
            
            # Convert to PIL for display/download
            inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else inpainted)
            
            with col2:
                st.image(inpainted_pil, caption="Watermark Removed", use_column_width=True)
                
                # Download button
                buf = io.BytesIO()
                inpainted_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Clean Image",
                    data=byte_im,
                    file_name="clean_image.png",
                    mime="image/png",
                    key="download-btn",
                    use_container_width=True,
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("""
    <div class="text-center mt-8 p-4 bg-gray-100 dark:bg-gray-800">
        <p class="text-sm">Built with ❤️ for educational purposes. <a href="https://github.com/yashrmusic/watermark" class="text-blue-500">View on GitHub</a></p>
    </div>
""", unsafe_allow_html=True)
