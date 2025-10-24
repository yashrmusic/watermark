import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import easyocr

# Initialize EasyOCR reader (run once)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

# Streamlit config
st.set_page_config(page_title="Gemini Watermark Remover", layout="wide")

# Custom CSS with Tailwind
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

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar p-4">', unsafe_allow_html=True)
    st.header("Settings & Support")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    inpaint_method = st.selectbox("Inpaint Method", ["TELEA (Fast)", "NS (Better Blend)"])
    st.markdown("""
        <h3 class="text-lg font-semibold">Support Us</h3>
        <p class="text-sm">[Your AdSense Ad Here]</p>
        <p class="text-xs text-gray-500">Disclaimer: Watermark removal may violate terms of service. Use responsibly.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Dark mode
if theme == "Dark":
    st.markdown('<style>body { background-color: #1f2937; color: white; }</style>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Drag and drop or browse (PNG/JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    st.subheader("Manual Watermark Selection")
    use_manual = st.checkbox("Manually select watermark area (recommended if auto fails)")
    if use_manual:
        x_start = st.slider("X Start", 0, 1000, 0)
        y_start = st.slider("Y Start", 0, 1000, 0)
        width = st.slider("Width", 10, 200, 50)
        height = st.slider("Height", 10, 200, 30)

if uploaded_file:
    with st.spinner("Processing image..."):
        # Load image
        image = Image.open(uploaded_file)
        # Convert PIL's RGB image to NumPy array
        img_array_rgb = np.array(image)
        
        # Fix: Convert RGB to BGR for OpenCV
        if len(img_array_rgb.shape) == 3:
            img_array = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        else:
            img_array = img_array_rgb  # Grayscale is fine
        
        img_height, img_width = img_array.shape[:2]
        
        with col1:
            # Display original RGB image
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Initialize mask
        mask = np.zeros((img_height, img_width), np.uint8)
        
        if use_manual:
            # Manual mask
            cv2.rectangle(mask, (x_start, y_start), (x_start + width, y_start + height), 255, -1)
            st.success(f"Manual mask applied: {width}x{height} at ({x_start}, {y_start})")
        else:
            # Preprocess for OCR (CLAHE to enhance contrast)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            else:
                gray = img_array
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # Auto-detect with lower confidence
            reader = load_ocr_reader()
            results = reader.readtext(gray)
            watermark_detected = False
            for (bbox, text, conf) in results:
                if 'ai' in text.lower() and conf > 0.3:
                    x, y = int(bbox[0][0]), int(bbox[0][1])
                    w, h = int(bbox[1][0] - bbox[0][0]), int(bbox[1][1] - bbox[0][1])
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    watermark_detected = True
                    st.success(f"Detected 'ai' at ({x}, {y}) with conf {conf:.2f}")
                    break
            
            # Larger fallback
            if not watermark_detected:
                st.warning("No 'ai' detected. Using larger bottom-right fallback.")
                x_start, y_start = img_width - 50, img_height - 30
                cv2.rectangle(mask, (x_start, y_start), (img_width, img_height), 255, -1)
        
        # Preview mask (red overlay)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_overlay = cv2.addWeighted(img_array, 0.8, mask_colored, 0.2, 0)
        st.image(Image.fromarray(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)), caption="Mask Preview (Red Overlay)", use_column_width=True)
        
        # Inpaint method
        method = cv2.INPAINT_TELEA if inpaint_method == "TELEA (Fast)" else cv2.INPAINT_NS
        
        try:
            if len(img_array.shape) == 3:  # Color image (BGR)
                b, g, r = cv2.split(img_array)
                b_inp = cv2.inpaint(b, mask, 5, method)
                g_inp = cv2.inpaint(g, mask, 5, method)
                r_inp = cv2.inpaint(r, mask, 5, method)
                inpainted = cv2.merge([b_inp, g_inp, r_inp])
            else:  # Grayscale
                inpainted = cv2.inpaint(img_array, mask, 5, method)
            
            # Convert BGR to RGB for display/download
            inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else inpainted)
            
            with col2:
                st.image(inpainted_pil, caption="Watermark Removed", use_column_width=True)
                st.success("Processing complete! Check the preview—use manual mode if needed.")
                
                # Download
                buf = io.BytesIO()
                inpainted_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Clean Image",
                    data=byte_im,
                    file_name="clean_image.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error processing: {str(e)}")

# Footer
st.markdown("""
    <div class="text-center mt-8 p-4 bg-gray-100 dark:bg-gray-800">
        <p class="text-sm">Built with ❤️ for educational purposes. <a href="https://github.com/yashrmusic/watermark" class="text-blue-500">View on GitHub</a></p>
    </div>
""", unsafe_allow_html=True)
