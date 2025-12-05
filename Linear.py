import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from fpdf import FPDF
import io
import tempfile

# ==========================================
# 1. KONFIGURASI & LOKALISASI (MULTI-LANGUAGE)
# ==========================================
st.set_page_config(layout="wide", page_title="2D Matrix Transformations", page_icon="📐")

# Dictionary untuk Multi-bahasa
LANGUAGES = {
    "English": {
        "title": "2D Matrix Transformations Explorer",
        "sidebar_title": "Settings",
        "select_lang": "Select Language",
        "transform_type": "Transformation Type",
        "params": "Parameters",
        "matrix_label": "Transformation Matrix",
        "points_label": "Points Data",
        "download_section": "Download Report",
        "orig_shape": "Original Shape",
        "trans_shape": "Transformed Shape",
        "types": ["Translation", "Rotation", "Scale", "Shear", "Reflection"],
        "rotation_deg": "Rotation Angle (degrees)",
        "scale_x": "Scale X",
        "scale_y": "Scale Y",
        "shear_x": "Shear X",
        "shear_y": "Shear Y",
        "reflect_axis": "Reflection Axis",
        "trans_x": "Translate X",
        "trans_y": "Translate Y",
        "download_pdf": "Download Report (PDF)",
        "download_csv": "Download Points (CSV)",
        "download_img": "Download Plot (PNG)",
        "reset": "Reset Transformations"
    },
    "Indonesia": {
        "title": "Eksplorasi Transformasi Matriks 2D",
        "sidebar_title": "Pengaturan",
        "select_lang": "Pilih Bahasa",
        "transform_type": "Jenis Transformasi",
        "params": "Parameter",
        "matrix_label": "Matriks Transformasi",
        "points_label": "Data Titik",
        "download_section": "Unduh Laporan",
        "orig_shape": "Bentuk Asli",
        "trans_shape": "Bentuk Transformasi",
        "types": ["Translasi", "Rotasi", "Skala", "Shear (Geser)", "Refleksi"],
        "rotation_deg": "Sudut Rotasi (derajat)",
        "scale_x": "Skala X",
        "scale_y": "Skala Y",
        "shear_x": "Shear X",
        "shear_y": "Shear Y",
        "reflect_axis": "Sumbu Refleksi",
        "trans_x": "Geser X (Translasi)",
        "trans_y": "Geser Y (Translasi)",
        "download_pdf": "Unduh Laporan (PDF)",
        "download_csv": "Unduh Data (CSV)",
        "download_img": "Unduh Grafik (PNG)",
        "reset": "Reset Transformasi"
    }
}

# ==========================================
# 2. LOGIKA MATEMATIKA (NUMPY)
# ==========================================

def get_arrow_shape():
    """Mengembalikan array NumPy (3, N) untuk bentuk panah asimetris dalam koordinat homogen."""
    # Format: [x, y, 1] (Homogeneous Coordinates)
    # Bentuk Panah: Ekor, Badan, Sayap Kanan, Ujung, Sayap Kiri, Badan, Ekor
    points = np.array([
        [0, 1, 1],  # Ekor Bawah
        [2, 1, 1],  # Badan Bawah
        [2, 0, 1],  # Sayap Bawah
        [4, 2, 1],  # Ujung Panah
        [2, 4, 1],  # Sayap Atas
        [2, 3, 1],  # Badan Atas
        [0, 3, 1],  # Ekor Atas
        [0, 1, 1]   # Kembali ke awal (menutup loop)
    ])
    # Transpose agar menjadi matriks 3xN untuk perkalian matriks
    return points.T 

def get_translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def get_rotation_matrix(degrees):
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def get_scale_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

def get_shear_matrix(kx, ky):
    return np.array([
        [1, kx, 0],
        [ky, 1, 0],
        [0, 0, 1]
    ])

def get_reflection_matrix(axis):
    if axis == 'X':
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == 'Y':
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else: # Origin/Both
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

# ==========================================
# 3. KELAS PDF GENERATOR
# ==========================================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, '2D Matrix Transformation Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(original_matrix, final_matrix, transformation_matrix, fig_img_bytes, lang_dict):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # 1. Info Matriks Transformasi Final
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Final Transformation Matrix (3x3 Homogeneous):", 0, 1)
    pdf.set_font("Courier", size=10)
    
    # Format matriks ke string rapi
    mat_str = ""
    for row in transformation_matrix:
        row_str = "  ".join([f"{x:6.2f}" for x in row])
        mat_str += f"| {row_str} |\n"
    
    pdf.multi_cell(0, 5, mat_str)
    pdf.ln(5)

    # 2. Gambar Grafik
    # Simpan bytes gambar ke file sementara agar FPDF bisa membacanya
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(fig_img_bytes)
        tmp_path = tmp_file.name

    pdf.image(tmp_path, x=10, y=pdf.get_y(), w=190) # Lebar A4 approx 210mm
    pdf.ln(100) # Geser ke bawah setelah gambar (sesuaikan dengan tinggi gambar)

    # 3. Data Point Summary (Table)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_y(180) # Pastikan di bawah gambar
    pdf.cell(0, 10, "Coordinate Data Summary:", 0, 1)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(30, 10, "Index", 1)
    pdf.cell(45, 10, "Original (x, y)", 1)
    pdf.cell(45, 10, "Transformed (x, y)", 1)
    pdf.ln()

    pdf.set_font("Arial", size=10)
    # Hanya ambil 8 titik pertama agar muat satu halaman contoh
    orig_pts = original_matrix[:2, :].T
    trans_pts = final_matrix[:2, :].T
    
    for i in range(len(orig_pts)):
        pdf.cell(30, 8, f"P{i+1}", 1)
        pdf.cell(45, 8, f"({orig_pts[i,0]:.2f}, {orig_pts[i,1]:.2f})", 1)
        pdf.cell(45, 8, f"({trans_pts[i,0]:.2f}, {trans_pts[i,1]:.2f})", 1)
        pdf.ln()

    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 4. MAIN APPLICATION
# ==========================================

def main():
    # Sidebar: Pilih Bahasa
    lang_key = st.sidebar.selectbox("Language / Bahasa", ["English", "Indonesia"])
    txt = LANGUAGES[lang_key]

    st.title(txt['title'])

    # --- STATE MANAGEMENT ---
    # Kita menggunakan matriks identitas sebagai base
    if 'current_matrix' not in st.session_state:
        st.session_state.current_matrix = np.eye(3)

    # --- SIDEBAR CONTROLS ---
    st.sidebar.title(txt['sidebar_title'])
    
    # Pilih Transformasi
    trans_type = st.sidebar.selectbox(txt['transform_type'], txt['types'])

    # Input Dinamis berdasarkan pilihan
    matrix_op = np.eye(3) # Operasi saat ini (default Identity)

    if trans_type in ["Translasi", "Translation"]:
        tx = st.sidebar.slider(txt['trans_x'], -5.0, 5.0, 0.0, 0.5)
        ty = st.sidebar.slider(txt['trans_y'], -5.0, 5.0, 0.0, 0.5)
        matrix_op = get_translation_matrix(tx, ty)

    elif trans_type in ["Rotasi", "Rotation"]:
        deg = st.sidebar.slider(txt['rotation_deg'], -180, 180, 0)
        matrix_op = get_rotation_matrix(deg)

    elif trans_type in ["Skala", "Scale"]:
        sx = st.sidebar.number_input(txt['scale_x'], value=1.0, step=0.1)
        sy = st.sidebar.number_input(txt['scale_y'], value=1.0, step=0.1)
        matrix_op = get_scale_matrix(sx, sy)

    elif trans_type in ["Shear (Geser)", "Shear"]:
        kx = st.sidebar.slider(txt['shear_x'], -2.0, 2.0, 0.0, 0.1)
        ky = st.sidebar.slider(txt['shear_y'], -2.0, 2.0, 0.0, 0.1)
        matrix_op = get_shear_matrix(kx, ky)

    elif trans_type in ["Refleksi", "Reflection"]:
        axis = st.sidebar.radio(txt['reflect_axis'], ['X', 'Y', 'Origin'])
        matrix_op = get_reflection_matrix(axis)

    # Tombol Reset
    if st.sidebar.button(txt['reset']):
        # Reset input sliders secara manual agak tricky di Streamlit, 
        # tapi kita bisa mereset matriks finalnya.
        # Di sini kita anggap user memanipulasi 'live' transformasi.
        pass 
        
    # --- PROSES TRANSFORMASI ---
    # 1. Ambil Bentuk Asli
    orig_shape = get_arrow_shape() # 3xN Matrix

    # 2. Terapkan Matriks Pilihan User
    # Dalam demo ini, kita menerapkan single transformation langsung ke original
    # (Untuk chaining complex, kita butuh list operasi, tapi ini demo interaktif langsung)
    final_transform_matrix = matrix_op
    
    # Hitung posisi baru: New = Matrix @ Old
    transformed_shape = final_transform_matrix @ orig_shape

    # --- VISUALISASI (PLOTLY) ---
    col1, col2 = st.columns([3, 1])

    with col1:
        fig = go.Figure()

        # Bentuk Asli (Abu-abu putus-putus)
        fig.add_trace(go.Scatter(
            x=orig_shape[0, :], 
            y=orig_shape[1, :],
            fill=None,
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name=txt['orig_shape']
        ))

        # Bentuk Transformasi (Warna Solid)
        fig.add_trace(go.Scatter(
            x=transformed_shape[0, :], 
            y=transformed_shape[1, :],
            fill='toself', # Mengisi poligon
            mode='lines+markers',
            line=dict(color='#636EFA', width=3),
            name=txt['trans_shape']
        ))

        # Atur Layout agar Cartesian tetap proporsional
        max_range = 10
        fig.update_layout(
            xaxis=dict(range=[-max_range, max_range], zeroline=True, zerolinewidth=2, title='X'),
            yaxis=dict(range=[-max_range, max_range], zeroline=True, zerolinewidth=2, title='Y', scaleanchor="x", scaleratio=1),
            height=600,
            showlegend=True,
            # Tema akan mengikuti dark/light mode Streamlit secara default jika template='streamlit'
            # tapi kita paksa sedikit agar grid terlihat jelas
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        # Deteksi Dark Mode sederhana (Streamlit native theme handling is automatic, 
        # but Plotly needs a hint sometimes. Let's stick to standard plotly_white for clarity on PDF)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Matrix")
        
        # Tampilkan Matriks dalam LaTeX
        st.latex(r'''
        M = \begin{bmatrix} 
        ''' + f"{final_transform_matrix[0,0]:.2f} & {final_transform_matrix[0,1]:.2f} & {final_transform_matrix[0,2]:.2f} \\\\" + 
        f"{final_transform_matrix[1,0]:.2f} & {final_transform_matrix[1,1]:.2f} & {final_transform_matrix[1,2]:.2f} \\\\" + 
        f"{final_transform_matrix[2,0]:.2f} & {final_transform_matrix[2,1]:.2f} & {final_transform_matrix[2,2]:.2f}" + 
        r'''
        \end{bmatrix}
        ''')
        
        st.info("Tip: M (3x3) applied to [x, y, 1]ᵀ")

        # Dataframe Preview
        df = pd.DataFrame({
            'Original X': orig_shape[0, :],
            'Original Y': orig_shape[1, :],
            'Transformed X': transformed_shape[0, :],
            'Transformed Y': transformed_shape[1, :]
        })
        with st.expander(txt['points_label']):
            st.dataframe(df, height=200)

    # --- DOWNLOAD SECTION ---
    st.divider()
    st.subheader(txt['download_section'])

    d_col1, d_col2, d_col3 = st.columns(3)

    # 1. Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    d_col1.download_button(
        label=f"📄 {txt['download_csv']}",
        data=csv,
        file_name='matrix_points.csv',
        mime='text/csv',
    )

    # 2. Download Image (PNG)
    # Kita butuh convert plotly fig ke bytes
    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
    d_col2.download_button(
        label=f"🖼️ {txt['download_img']}",
        data=img_bytes,
        file_name='transformation_plot.png',
        mime='image/png'
    )

    # 3. Download PDF (Complete Report)
    # Generate PDF saat tombol ditekan (menggunakan cache data jika berat)
    try:
        pdf_bytes = create_pdf(orig_shape, transformed_shape, final_transform_matrix, img_bytes, txt)
        d_col3.download_button(
            label=f"📕 {txt['download_pdf']}",
            data=pdf_bytes,
            file_name='transformation_report.pdf',
            mime='application/pdf'
        )
    except Exception as e:
        d_col3.error(f"Error generating PDF: {e}")
        d_col3.caption("Pastikan 'kaleido' terinstall di environment.")

if __name__ == "__main__":
    main()