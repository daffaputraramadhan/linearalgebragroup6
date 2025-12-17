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
