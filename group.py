import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

# ===================== 1. CONFIG (HANYA BOLEH SEKALI DI AWAL) =====================
st.set_page_config(
    page_title="Explanation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== 2. TRANSLATIONS DATA (SANGAT PENTING) =====================
# Ini adalah bagian yang sebelumnya hilang dan menyebabkan error.
translations = {
    "id": {
        "title": "Matrix Transformation & Image Processing",
        "subtitle": "Eksplorasi interaktif Aljabar Linear dalam manipulasi gambar digital",
        "app_goal": "### 🎯 Tujuan Aplikasi\nAplikasi ini dibuat untuk memvisualisasikan bagaimana **Operasi Matriks** digunakan secara nyata dalam teknologi pengolahan citra digital sehari-hari.",
        "features": "**Fitur Utama:** Translasi, Rotasi, Scaling, dan Filtering (Blur/Edge Detection).",
        "concept_1_title": "📐 Transformasi Geometri",
        "concept_1_text1": "Mengubah posisi (Translasi), ukuran (Scaling), atau orientasi (Rotasi) gambar.",
        "concept_1_text2": "Menggunakan perkalian matriks 3x3 (Affine Transform).",
        "concept_2_title": "🎨 Filtering Citra",
        "concept_2_text1": "Memanipulasi nilai pixel untuk efek Blur, Sharpen, atau Deteksi Tepi.",
        "concept_2_text2": "Menggunakan operasi Konvolusi dengan matriks Kernel.",
        "concept_3_title": "🖼️ Segmentasi Objek",
        "concept_3_text1": "Memisahkan objek depan (Foreground) dari latar belakang (Background).",
        "concept_3_text2": "Menggunakan operasi matriks pada channel warna (Thresholding).",
        "quick_concepts": "💡 Konsep Singkat",
        "quick_concepts_text": "Setiap gambar digital adalah sekumpulan matriks angka. Dengan memanipulasi angka tersebut menggunakan Aljabar Linear, kita dapat mengubah tampilan gambar.",
        "upload_title": "📂 Unggah Gambar",
        "upload_label": "Pilih gambar (PNG/JPG/JPEG)",
        "upload_success": "Gambar berhasil dimuat!",
        "upload_preview": "Gambar Asli",
        "upload_info": "Silakan unggah gambar untuk memulai eksperimen.",
        "tools_title": "🛠️ Laboratorium Matriks",
        "tools_subtitle": "Pilih transformasi di bawah ini",
        "geo_title": "1. Transformasi Geometri (Affine)",
        "geo_desc": "Memindahkan koordinat pixel menggunakan perkalian matriks.",
        "geo_info": "Unggah gambar terlebih dahulu untuk menggunakan fitur ini.",
        "btn_translation": "Translasi (Geser)",
        "btn_scaling": "Scaling (Ubah Ukuran)",
        "btn_rotation": "Rotasi (Putar)",
        "btn_shearing": "Shearing (Miringkan)",
        "btn_reflection": "Refleksi (Cermin)",
        "btn_apply": "Terapkan",
        "trans_settings": "**Pengaturan Translasi**",
        "trans_dx": "Geser X (Horizontal)",
        "trans_dy": "Geser Y (Vertikal)",
        "trans_result": "Hasil Translasi",
        "scale_settings": "**Pengaturan Scaling**",
        "scale_x": "Skala X (Lebar)",
        "scale_y": "Skala Y (Tinggi)",
        "scale_result": "Hasil Scaling",
        "rot_settings": "**Pengaturan Rotasi**",
        "rot_angle": "Sudut Rotasi (Derajat)",
        "rot_result": "Hasil Rotasi",
        "shear_settings": "**Pengaturan Shearing**",
        "shear_x": "Geser Sumbu X",
        "shear_y": "Geser Sumbu Y",
        "shear_result": "Hasil Shearing",
        "refl_settings": "**Pengaturan Refleksi**",
        "refl_axis": "Sumbu Cermin",
        "refl_result": "Hasil Refleksi",
        "axis_x": "Sumbu X (Balik Vertikal)",
        "axis_y": "Sumbu Y (Balik Horizontal)",
        "axis_diag": "Diagonal",
        "hist_title": "📊 Histogram Warna",
        "hist_desc": "Melihat distribusi intensitas warna Merah, Hijau, dan Biru dalam matriks gambar.",
        "btn_histogram": "Tampilkan Histogram",
        "hist_warning": "Unggah gambar terlebih dahulu.",
        "filter_title": "2. Filtering & Konvolusi",
        "filter_desc": "Mengubah nilai pixel berdasarkan tetangganya (Kernel Matrix).",
        "filter_info": "Unggah gambar untuk mencoba filter.",
        "btn_blur": "Blur (Kabur)",
        "btn_sharpen": "Sharpen (Tajam)",
        "btn_background": "Hapus Background",
        "btn_grayscale": "Grayscale",
        "btn_edge": "Deteksi Tepi",
        "btn_brightness": "Kecerahan",
        "blur_settings": "**Pengaturan Blur**",
        "blur_kernel": "Ukuran Kernel (Ganjil)",
        "blur_result": "Hasil Blur",
        "sharpen_settings": "**Pengaturan Sharpen**",
        "sharpen_desc": "Meningkatkan kontras pada tepi objek.",
        "sharpen_result": "Hasil Sharpening",
        "bg_settings": "**Pengaturan Background**",
        "bg_method": "Metode Penghapusan",
        "bg_result": "Hasil Hapus Background",
        "gray_settings": "**Pengaturan Grayscale**",
        "gray_desc": "Mengubah citra RGB (3 channel) menjadi 1 channel intensitas.",
        "gray_result": "Hasil Grayscale",
        "edge_settings": "**Pengaturan Deteksi Tepi**",
        "edge_method": "Metode Algoritma",
        "edge_result": "Hasil Deteksi Tepi",
        "bright_settings": "**Kecerahan & Kontras**",
        "bright_brightness": "Tingkat Kecerahan",
        "bright_contrast": "Tingkat Kontras",
        "bright_result": "Hasil Penyesuaian",
        "team_title": "👥 Anggota Tim",
        "team_subtitle": "Project Kelompok - Aljabar Linear",
        "team_sid": "NIM:",
        "team_role": "Peran:",
        "team_group": "Kelompok:",
        "team_Contribution": "Kontribusi:"
    },
    "en": {
        "title": "Matrix Transformation & Image Processing",
        "subtitle": "Interactive exploration of Linear Algebra in digital image manipulation",
        "app_goal": "### 🎯 App Goal\nTo visualize how **Matrix Operations** are practically used in everyday digital image processing technologies.",
        "features": "**Key Features:** Translation, Rotation, Scaling, and Filtering (Blur/Edge Detection).",
        "concept_1_title": "📐 Geometric Transform",
        "concept_1_text1": "Changing position (Translation), size (Scaling), or orientation (Rotation).",
        "concept_1_text2": "Uses 3x3 Matrix Multiplication (Affine Transform).",
        "concept_2_title": "🎨 Image Filtering",
        "concept_2_text1": "Manipulating pixel values for Blur, Sharpen, or Edge Detection effects.",
        "concept_2_text2": "Uses Convolution operations with Kernel matrices.",
        "concept_3_title": "🖼️ Object Segmentation",
        "concept_3_text1": "Separating Foreground objects from the Background.",
        "concept_3_text2": "Uses matrix operations on color channels (Thresholding).",
        "quick_concepts": "💡 Quick Concept",
        "quick_concepts_text": "Every digital image is a matrix of numbers. By manipulating these numbers using Linear Algebra, we modify the image appearance.",
        "upload_title": "📂 Upload Image",
        "upload_label": "Choose an image (PNG/JPG/JPEG)",
        "upload_success": "Image loaded successfully!",
        "upload_preview": "Original Image",
        "upload_info": "Please upload an image to start experimenting.",
        "tools_title": "🛠️ Matrix Laboratory",
        "tools_subtitle": "Choose a transformation below",
        "geo_title": "1. Geometric Transformation (Affine)",
        "geo_desc": "Moving pixel coordinates using matrix multiplication.",
        "geo_info": "Upload an image first to use this feature.",
        "btn_translation": "Translation",
        "btn_scaling": "Scaling",
        "btn_rotation": "Rotation",
        "btn_shearing": "Shearing",
        "btn_reflection": "Reflection",
        "btn_apply": "Apply",
        "trans_settings": "**Translation Settings**",
        "trans_dx": "Shift X (Horizontal)",
        "trans_dy": "Shift Y (Vertical)",
        "trans_result": "Translation Result",
        "scale_settings": "**Scaling Settings**",
        "scale_x": "Scale X (Width)",
        "scale_y": "Scale Y (Height)",
        "scale_result": "Scaling Result",
        "rot_settings": "**Rotation Settings**",
        "rot_angle": "Rotation Angle (Degrees)",
        "rot_result": "Rotation Result",
        "shear_settings": "**Shearing Settings**",
        "shear_x": "Shear X Axis",
        "shear_y": "Shear Y Axis",
        "shear_result": "Shearing Result",
        "refl_settings": "**Reflection Settings**",
        "refl_axis": "Mirror Axis",
        "refl_result": "Reflection Result",
        "axis_x": "X Axis (Flip Vertical)",
        "axis_y": "Y Axis (Flip Horizontal)",
        "axis_diag": "Diagonal",
        "hist_title": "📊 Color Histogram",
        "hist_desc": "View the distribution of Red, Green, and Blue intensity in the image matrix.",
        "btn_histogram": "Show Histogram",
        "hist_warning": "Please upload an image first.",
        "filter_title": "2. Filtering & Convolution",
        "filter_desc": "Changing pixel values based on neighbors (Kernel Matrix).",
        "filter_info": "Upload an image to try filters.",
        "btn_blur": "Blur",
        "btn_sharpen": "Sharpen",
        "btn_background": "Remove Background",
        "btn_grayscale": "Grayscale",
        "btn_edge": "Edge Detection",
        "btn_brightness": "Brightness",
        "blur_settings": "**Blur Settings**",
        "blur_kernel": "Kernel Size (Odd)",
        "blur_result": "Blur Result",
        "sharpen_settings": "**Sharpen Settings**",
        "sharpen_desc": "Enhancing contrast at object edges.",
        "sharpen_result": "Sharpening Result",
        "bg_settings": "**Background Settings**",
        "bg_method": "Removal Method",
        "bg_result": "Background Removal Result",
        "gray_settings": "**Grayscale Settings**",
        "gray_desc": "Converting RGB image (3 channels) to 1 intensity channel.",
        "gray_result": "Grayscale Result",
        "edge_settings": "**Edge Detection Settings**",
        "edge_method": "Algorithm Method",
        "edge_result": "Edge Detection Result",
        "bright_settings": "**Brightness & Contrast**",
        "bright_brightness": "Brightness Level",
        "bright_contrast": "Contrast Level",
        "bright_result": "Adjustment Result",
        "team_title": "👥 Team Members",
        "team_subtitle": "Group Project - Linear Algebra",
        "team_sid": "Student ID:",
        "team_role": "Role:",
        "team_group": "Group:",
        "team_Contribution": "Contribution:"
    }
}

# ===================== 3. HELPER FUNCTIONS =====================

def set_video_background(video_path: str):
    """Set an mp4 video as full-screen background using HTML/CSS."""
    if not os.path.exists(video_path):
        # Silent fail or use a color if video missing to prevent crash
        return

    with open(video_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    video_data_url = f"data:video/mp4;base64,{b64}"

    st.markdown(
        f"""
        <style>
        .video-bg {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
            opacity: 0.8; /* Sedikit transparan agar tulisan terbaca */
        }}
        .stApp {{
            background: transparent !important;
        }}
        </style>
        <video class="video-bg" autoplay muted loop playsinline>
            <source src="{video_data_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

def load_image(file):
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)
    return img_np

def to_opencv(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def to_streamlit(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_affine_transform(img_rgb, M, output_size=None):
    img_bgr = to_opencv(img_rgb)
    h, w = img_bgr.shape[:2]
    if output_size is None:
        output_size = (w, h)

    if M.shape == (3, 3):
        M_affine = M[0:2, :]
    else:
        M_affine = M

    transformed = cv2.warpAffine(
        img_bgr, M_affine, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return to_streamlit(transformed)

def manual_convolution_gray(img_gray, kernel):
    # Optimasi sederhana menggunakan filter2D OpenCV untuk performa
    # Jika ingin full manual loop (lambat di Python), bisa dikembalikan
    return cv2.filter2D(img_gray, -1, kernel)

def rgb_to_gray(img_rgb):
    img_bgr = to_opencv(img_rgb)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray

def adjust_brightness_contrast(img_rgb, brightness=0, contrast=0):
    img_bgr = to_opencv(img_rgb)
    beta = brightness
    alpha = 1 + (contrast / 100.0)
    adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return to_streamlit(adjusted)

def image_to_bytes(img_rgb, fmt="PNG"):
    """Convert numpy RGB image to bytes for download."""
    pil_img = Image.fromarray(img_rgb.astype("uint8"))
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def safe_display_square_image(path):
    """Display image in square format with proper cropping"""
    if os.path.exists(path):
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img_cropped = img.crop((left, top, right, bottom))
            img_resized = img_cropped.resize((140, 140), Image.Resampling.LANCZOS)

            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            st.markdown(f"""
            <div class="crystal-shape">
                <img src="data:image/jpeg;base64,{img_str}" alt="Team member"/>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        st.markdown("""
        <div class="team-photo-container">
            <div style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:#ddd; color:#666; font-size:12px;">
                No Image
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Background Removal Helpers (Simplified for Stability) ---
def segment_foreground(image: np.ndarray) -> np.ndarray:
    """Simple center ellipse mask for demo purposes"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.35), int(h * 0.45)) 
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask

def simple_background_removal_hsv(img_rgb):
    img_bgr = to_opencv(img_rgb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Deteksi warna hijau/biru (chroma key sederhana)
    lower = np.array([35, 40, 40]) # Range hijau kira-kira
    upper = np.array([85, 255, 255])
    mask_bg = cv2.inRange(hsv, lower, upper)
    mask_fg = cv2.bitwise_not(mask_bg)
    fg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_fg)
    
    # Buat alpha channel
    b, g, r = cv2.split(fg)
    rgba = [b, g, r, mask_fg]
    dst = cv2.merge(rgba, 4)
    
    return cv2.cvtColor(dst, cv2.COLOR_BGRA2RGBA)

def remove_background_advanced(image, mode="auto", output_mode="transparent", **kwargs):
    # Menggunakan dummy mask karena implementasi ML/AI berat untuk script ini
    raw_mask = segment_foreground(image)
    
    # Smooth mask
    raw_mask = cv2.GaussianBlur(raw_mask, (21, 21), 0)
    
    if output_mode == "transparent":
        rgba = np.dstack([image, raw_mask])
        return rgba
    
    return image

# ===================== 4. SESSION STATE INIT =====================

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "light"
if "language" not in st.session_state:
    st.session_state["language"] = "id"
if "original_img" not in st.session_state:
    st.session_state.original_img = None
if "geo_transform" not in st.session_state:
    st.session_state["geo_transform"] = None
if "image_filter" not in st.session_state:
    st.session_state["image_filter"] = None

# ===================== 5. UI SETUP & CSS =====================

# Panggil fungsi background video (pastikan file ada, kalau tidak, tidak error)
set_video_background("assets/background.mp4")

base_css = """
<style>
.block-container {
    max-width: 1200px;
    padding: 2rem 2rem 2rem 2rem;
}
section[data-testid="stExpander"]{
    border-radius:10px;
    padding:8px;
    box-shadow:0 1px 6px rgba(0,0,0,0.04);
}
div[data-testid="column"] button {
    width: 100%;
}
/* Simple rectangular container for photos and text */
.crystal-shape {
    width: 140px;
    height: 140px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f0f0f0;
    border: 3px solid #4CAF50;
    border-radius: 5px;
}
.crystal-shape img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 5px;
}
.crystal-text {
    padding: 10px;
    box-sizing: border-box;
    font-size: 12px;
    text-align: center;
    border: 1px solid #ccc;
    background: #fff;
}
</style>
"""

light_css = """
<style>
.stMarkdown, .stMarkdown p, .stMarkdown li, h1, h2, h3 {
    color: #000000 !important;
}
button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #4CAF50 !important;
}
.stContainer {
    background-color: rgba(255,255,255,0.9) !important;
}
</style>
"""

dark_css = """
<style>
.stMarkdown, .stMarkdown p, .stMarkdown li, h1, h2, h3 {
    color: #00FF00 !important;
}
button[kind="secondary"] {
    background-color: #1e3a1e !important;
    color: #c8e6c9 !important;
    border: 2px solid #66bb6a !important;
}
</style>
"""

st.markdown(base_css, unsafe_allow_html=True)
if st.session_state["theme_mode"] == "light":
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# ===================== 6. HEADER & NAV =====================

lang = st.session_state["language"]
t = translations[lang] # Load dictionary bahasa
theme_mode = st.session_state["theme_mode"]

with st.container(border=True):
    header_col1, header_col2, header_col3 = st.columns([4, 1, 1], vertical_alignment="center")

    with header_col1:
        st.title(t["title"])

    with header_col2:
        lang_button_text = "🇬🇧 EN" if lang == "id" else "🇮🇩 ID"
        if st.button(lang_button_text, key="lang_toggle", use_container_width=True):
            st.session_state["language"] = "en" if lang == "id" else "id"
            st.rerun()

    with header_col3:
        theme_button_text = "🌙 Dark" if theme_mode == "light" else "☀ Light"
        if st.button(theme_button_text, key="theme_toggle", use_container_width=True):
            st.session_state["theme_mode"] = "dark" if theme_mode == "light" else "light"
            st.rerun()

st.subheader(t["subtitle"])

# ===================== 7. MAIN CONTENT =====================

with st.container(border=True):
    st.markdown(t["app_goal"])
    st.markdown(t["features"])

# --- Concept Boxes ---
col1, col2, col3 = st.columns(3, vertical_alignment="top")

with col1:
    with st.container(border=True):
        st.markdown(f"""
        <div class="crystal-text">
            <strong>{t['concept_1_title']}</strong><br>
            {t["concept_1_text1"]}<br>
            {t["concept_1_text2"]}
        </div>
        """, unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown(f"""
        <div class="crystal-text">
            <strong>{t['concept_2_title']}</strong><br>
            {t["concept_2_text1"]}<br>
            {t["concept_2_text2"]}
        </div>
        """, unsafe_allow_html=True)

with col3:
    with st.container(border=True):
        st.markdown(f"""
        <div class="crystal-text">
            <strong>{t['concept_3_title']}</strong><br>
            {t["concept_3_text1"]}<br>
            {t["concept_3_text2"]}
        </div>
        """, unsafe_allow_html=True)

# --- Concepts Reminder ---
with st.container(border=True):
    st.markdown(f"""
    <div class="crystal-text">
        <strong>{t['quick_concepts']}</strong><br>
        {t['quick_concepts_text']}
    </div>
    """, unsafe_allow_html=True)

# ===================== 8. NAVIGATION TABS =====================

# Create tabs for navigation
tab_upload, tab_geometry, tab_filter, tab_team = st.tabs([
    "📤 Upload Image", 
    "📐 Geometric Transform", 
    "🎨 Filter & Convolution", 
    "👥 Team"
])

# ===================== 9. TAB 1: UPLOAD IMAGE =====================
with tab_upload:
    with st.container(border=True):
        st.markdown(f"### {t['upload_title']}")
        
        uploaded_file = st.file_uploader(
            t["upload_label"], 
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        if uploaded_file is not None:
            try:
                original_img = load_image(uploaded_file)
                st.session_state.original_img = original_img
                st.success(t["upload_success"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption=t["upload_preview"], use_column_width=True)
                with col2:
                    st.info(f"**Image Info:**")
                    st.write(f"Size: {original_img.shape[1]} x {original_img.shape[0]}")
                    st.write(f"Channels: {original_img.shape[2] if len(original_img.shape) > 2 else 1}")
                    
                    # Quick histogram preview
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ('r', 'g', 'b')
                    for i, color in enumerate(colors):
                        histogram = cv2.calcHist([to_opencv(original_img)], [i], None, [256], [0, 256])
                        ax.plot(histogram, color=color, alpha=0.7)
                    ax.set_xlim([0, 256])
                    ax.set_xlabel('Pixel Intensity')
                    ax.set_ylabel('Frequency')
                    ax.set_title('RGB Histogram Preview')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        else:
            st.info(t["upload_info"])
            # Display placeholder image
            placeholder = np.ones((300, 400, 3), dtype=np.uint8) * 240
            st.image(placeholder, caption="No image uploaded", use_column_width=True)

# ===================== 10. TAB 2: GEOMETRIC TRANSFORMATIONS =====================
with tab_geometry:
    with st.container(border=True):
        st.markdown(f"### {t['geo_title']}")
        st.markdown(t["geo_desc"])
        
        if st.session_state.original_img is None:
            st.warning(t["geo_info"])
            st.stop()
        
        original_img = st.session_state.original_img
        
        # Transformation selection buttons
        col_btns = st.columns(5)
        with col_btns[0]:
            translation_btn = st.button(t["btn_translation"], use_container_width=True)
        with col_btns[1]:
            scaling_btn = st.button(t["btn_scaling"], use_container_width=True)
        with col_btns[2]:
            rotation_btn = st.button(t["btn_rotation"], use_container_width=True)
        with col_btns[3]:
            shearing_btn = st.button(t["btn_shearing"], use_container_width=True)
        with col_btns[4]:
            reflection_btn = st.button(t["btn_reflection"], use_container_width=True)
        
        # Set active transform based on button click
        if translation_btn:
            st.session_state["geo_transform"] = "translation"
        elif scaling_btn:
            st.session_state["geo_transform"] = "scaling"
        elif rotation_btn:
            st.session_state["geo_transform"] = "rotation"
        elif shearing_btn:
            st.session_state["geo_transform"] = "shearing"
        elif reflection_btn:
            st.session_state["geo_transform"] = "reflection"
        
        # If no transform selected yet, default to translation
        if st.session_state["geo_transform"] is None:
            st.session_state["geo_transform"] = "translation"
        
        current_transform = st.session_state["geo_transform"]
        
        # Display transformation controls
        st.markdown(f"**Selected:** {current_transform.upper()}")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Transformation parameters
            if current_transform == "translation":
                st.markdown(t["trans_settings"])
                dx = st.slider(t["trans_dx"], -200, 200, 0, 10)
                dy = st.slider(t["trans_dy"], -200, 200, 0, 10)
                
                # Create translation matrix
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                
                if st.button(t["btn_apply"], key="apply_trans"):
                    result_img = apply_affine_transform(original_img, M)
                    st.session_state.last_result = result_img
                    
            elif current_transform == "scaling":
                st.markdown(t["scale_settings"])
                scale_x = st.slider(t["scale_x"], 0.1, 3.0, 1.0, 0.1)
                scale_y = st.slider(t["scale_y"], 0.1, 3.0, 1.0, 0.1)
                
                h, w = original_img.shape[:2]
                new_w, new_h = int(w * scale_x), int(h * scale_y)
                
                M = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
                
                if st.button(t["btn_apply"], key="apply_scale"):
                    result_img = apply_affine_transform(original_img, M, (new_w, new_h))
                    st.session_state.last_result = result_img
                    
            elif current_transform == "rotation":
                st.markdown(t["rot_settings"])
                angle = st.slider(t["rot_angle"], -180, 180, 0, 5)
                
                h, w = original_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                if st.button(t["btn_apply"], key="apply_rot"):
                    result_img = apply_affine_transform(original_img, M)
                    st.session_state.last_result = result_img
                    
            elif current_transform == "shearing":
                st.markdown(t["shear_settings"])
                shear_x = st.slider(t["shear_x"], -1.0, 1.0, 0.0, 0.1)
                shear_y = st.slider(t["shear_y"], -1.0, 1.0, 0.0, 0.1)
                
                M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
                
                if st.button(t["btn_apply"], key="apply_shear"):
                    result_img = apply_affine_transform(original_img, M)
                    st.session_state.last_result = result_img
                    
            elif current_transform == "reflection":
                st.markdown(t["refl_settings"])
                axis = st.radio(t["refl_axis"], 
                              [t["axis_x"], t["axis_y"], t["axis_diag"]],
                              horizontal=True)
                
                if axis == t["axis_x"]:
                    M = np.float32([[1, 0, 0], [0, -1, original_img.shape[0]]])
                elif axis == t["axis_y"]:
                    M = np.float32([[-1, 0, original_img.shape[1]], [0, 1, 0]])
                else:  # diagonal
                    M = np.float32([[0, 1, 0], [1, 0, 0]])
                
                if st.button(t["btn_apply"], key="apply_refl"):
                    result_img = apply_affine_transform(original_img, M)
                    st.session_state.last_result = result_img
        
        with col_right:
            # Display results
            if "last_result" in st.session_state:
                st.image(st.session_state.last_result, 
                        caption=t[f"{current_transform}_result"], 
                        use_column_width=True)
                
                # Download button
                img_bytes = image_to_bytes(st.session_state.last_result)
                st.download_button(
                    label="📥 Download Result",
                    data=img_bytes,
                    file_name=f"{current_transform}_result.png",
                    mime="image/png"
                )
            else:
                st.info("Apply transformation to see results here")

# ===================== 11. TAB 3: FILTER & CONVOLUTION =====================
with tab_filter:
    with st.container(border=True):
        st.markdown(f"### {t['filter_title']}")
        st.markdown(t["filter_desc"])
        
        if st.session_state.original_img is None:
            st.warning(t["filter_info"])
            st.stop()
        
        original_img = st.session_state.original_img
        
        # Filter selection buttons
        col_filters = st.columns([2, 2, 2, 2, 2, 2])
        filter_buttons = [
            t["btn_blur"], t["btn_sharpen"], t["btn_background"],
            t["btn_grayscale"], t["btn_edge"], t["btn_brightness"]
        ]
        
        for idx, (col, btn_text) in enumerate(zip(col_filters, filter_buttons)):
            with col:
                if st.button(btn_text, key=f"filter_btn_{idx}", use_container_width=True):
                    st.session_state["image_filter"] = btn_text.lower()
        
        # Default filter
        if st.session_state["image_filter"] is None:
            st.session_state["image_filter"] = "blur"
        
        current_filter = st.session_state["image_filter"]
        
        col_filter_left, col_filter_right = st.columns(2)
        
        with col_filter_left:
            # Filter controls
            if "blur" in current_filter:
                st.markdown(t["blur_settings"])
                kernel_size = st.slider(t["blur_kernel"], 3, 31, 5, 2)
                
                if st.button(t["btn_apply"], key="apply_blur"):
                    img_bgr = to_opencv(original_img)
                    blurred = cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), 0)
                    result_img = to_streamlit(blurred)
                    st.session_state.filter_result = result_img
                    
            elif "sharpen" in current_filter:
                st.markdown(t["sharpen_settings"])
                st.markdown(t["sharpen_desc"])
                
                # Sharpen kernel
                sharpen_kernel = np.array([
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                ])
                
                intensity = st.slider("Intensity", 1.0, 5.0, 2.0, 0.1)
                custom_kernel = np.array([
                    [0, -1, 0],
                    [-1, intensity, -1],
                    [0, -1, 0]
                ])
                
                if st.button(t["btn_apply"], key="apply_sharpen"):
                    img_bgr = to_opencv(original_img)
                    sharpened = cv2.filter2D(img_bgr, -1, custom_kernel)
                    result_img = to_streamlit(sharpened)
                    st.session_state.filter_result = result_img
                    
            elif "background" in current_filter:
                st.markdown(t["bg_settings"])
                method = st.selectbox(t["bg_method"], 
                                    ["Simple Ellipse Mask", "HSV Color Range"])
                
                if st.button(t["btn_apply"], key="apply_bg"):
                    if method == "Simple Ellipse Mask":
                        mask = segment_foreground(original_img)
                        result_img = cv2.bitwise_and(original_img, original_img, mask=mask)
                    else:
                        result_img = simple_background_removal_hsv(original_img)
                    st.session_state.filter_result = result_img
                    
            elif "grayscale" in current_filter:
                st.markdown(t["gray_settings"])
                st.markdown(t["gray_desc"])
                
                method = st.radio("Conversion Method", 
                                ["Average", "Weighted (Luminosity)", "OpenCV"])
                
                if st.button(t["btn_apply"], key="apply_gray"):
                    if method == "Average":
                        gray = np.mean(original_img, axis=2).astype(np.uint8)
                        result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    elif method == "Weighted (Luminosity)":
                        # ITU-R BT.601 standard
                        gray = np.dot(original_img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                        result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    else:
                        img_bgr = to_opencv(original_img)
                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    st.session_state.filter_result = result_img
                    
            elif "edge" in current_filter:
                st.markdown(t["edge_settings"])
                edge_method = st.selectbox(t["edge_method"], 
                                         ["Sobel", "Canny", "Laplacian"])
                
                if edge_method == "Canny":
                    threshold1 = st.slider("Threshold 1", 0, 255, 100)
                    threshold2 = st.slider("Threshold 2", 0, 255, 200)
                
                if st.button(t["btn_apply"], key="apply_edge"):
                    gray = rgb_to_gray(original_img)
                    
                    if edge_method == "Sobel":
                        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                        edges = cv2.magnitude(sobelx, sobely)
                        edges = np.uint8(np.absolute(edges))
                    elif edge_method == "Canny":
                        edges = cv2.Canny(gray, threshold1, threshold2)
                    else:  # Laplacian
                        edges = cv2.Laplacian(gray, cv2.CV_64F)
                        edges = np.uint8(np.absolute(edges))
                    
                    result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    st.session_state.filter_result = result_img
                    
            elif "brightness" in current_filter:
                st.markdown(t["bright_settings"])
                brightness = st.slider(t["bright_brightness"], -100, 100, 0)
                contrast = st.slider(t["bright_contrast"], -50, 50, 0)
                
                if st.button(t["btn_apply"], key="apply_bright"):
                    result_img = adjust_brightness_contrast(original_img, brightness, contrast)
                    st.session_state.filter_result = result_img
        
        with col_filter_right:
            # Display filter results
            if "filter_result" in st.session_state:
                st.image(st.session_state.filter_result, 
                        caption=t[f"{current_filter}_result"], 
                        use_column_width=True)
                
                # Download button
                img_bytes = image_to_bytes(st.session_state.filter_result)
                st.download_button(
                    label="📥 Download Result",
                    data=img_bytes,
                    file_name=f"{current_filter}_result.png",
                    mime="image/png"
                )
            else:
                st.info("Apply filter to see results here")

# ===================== 12. TAB 4: TEAM INFORMATION =====================
with tab_team:
    with st.container(border=True):
        st.markdown(f"### {t['team_title']}")
        st.markdown(t["team_subtitle"])
        
        # Team members data
        team_members = [
            {
                "name": "John Doe",
                "sid": "1234567890",
                "role": "Frontend Developer",
                "photo": "assets/team/john.jpg",  # Pastikan file ini ada
                "contribution": "UI/UX Design, Frontend Development"
            },
            {
                "name": "Jane Smith",
                "sid": "0987654321",
                "role": "Backend Developer",
                "photo": "assets/team/jane.jpg",
                "contribution": "Algorithm Implementation, Data Processing"
            },
            {
                "name": "Alex Johnson",
                "sid": "1122334455",
                "role": "Data Scientist",
                "photo": "assets/team/alex.jpg",
                "contribution": "Image Processing Algorithms, Documentation"
            },
            {
                "name": "Maria Garcia",
                "sid": "5566778899",
                "role": "Project Manager",
                "photo": "assets/team/maria.jpg",
                "contribution": "Project Coordination, Testing"
            }
        ]
        
        # Display team members in a grid
        cols = st.columns(4)
        for idx, (col, member) in enumerate(zip(cols, team_members)):
            with col:
                with st.container(border=True):
                    # Display photo
                    safe_display_square_image(member["photo"])
                    
                    # Member info
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px;'>
                        <h4 style='margin-bottom: 5px;'>{member['name']}</h4>
                        <p style='margin: 2px; font-size: 12px;'><strong>{t['team_sid']}</strong> {member['sid']}</p>
                        <p style='margin: 2px; font-size: 12px;'><strong>{t['team_role']}</strong> {member['role']}</p>
                        <p style='margin: 2px; font-size: 11px; color: #666;'><strong>{t['team_Contribution']}</strong> {member['contribution']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Project description
        st.markdown("---")
        st.markdown("""
        ### About This Project
        This interactive application demonstrates the practical application of Linear Algebra concepts 
        in digital image processing. It showcases how matrix operations are fundamental to modern 
        computer vision and graphics technologies.
        
        **Technologies Used:**
        - Streamlit for web interface
        - OpenCV for image processing
        - NumPy for matrix operations
        - Matplotlib for visualization
        
        **Linear Algebra Concepts Applied:**
        1. Matrix transformations (Affine transformations)
        2. Convolution operations
        3. Eigenvalues/vectors (implied in certain filters)
        4. Vector spaces (color spaces)
        """)

# ===================== 13. FOOTER =====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.markdown("**Linear Algebra Project** • Industrial Engineering Class 2")
with footer_col2:
    st.markdown(f"Language: {'🇮🇩 Indonesian' if lang == 'id' else '🇬🇧 English'}")
with footer_col3:
    st.markdown(f"Theme: {'🌙 Dark' if theme_mode == 'dark' else '☀ Light'}")
