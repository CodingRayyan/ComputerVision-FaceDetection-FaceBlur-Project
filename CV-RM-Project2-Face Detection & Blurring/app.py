import cv2
import mediapipe as mp
import tempfile
import streamlit as st
import numpy as np
import os
import time

# ---------------- Page Config ----------------
st.set_page_config(page_title="Face Blur App", page_icon="😎", layout="wide")

# ---------------- Header ----------------
st.markdown("""
    <h1 style='text-align: center; color: #FFD700;'>😎 Real-Time Face Blur App using MediaPipe + OpenCV</h1>
    <h3 style='text-align: center; color: #FFFFFF;'>👨‍💻 Developed by <span style='color: #00BFFF;'>Rayyan Ahmed</span></h3>
    <hr style='border: 1px solid #FFD700; width: 80%; margin: auto;'>
""", unsafe_allow_html=True)

# ---------------- Background ----------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)),
                      url("https://www.gorspa.org/wp-content/uploads/iStock-biometrics2-640x507.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Styling ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 65, 0, 0.4);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00BFFF; }
::-webkit-scrollbar-thumb { background: #FFD700; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Info ----------------
with st.sidebar.expander("📌 Project Intro"):
    st.markdown("""
    ### 🎯 **Project Goal**
    - Detect faces in **images, videos, or live webcam**  
    - Apply **blurring** to protect privacy  
    - Compare **original vs blurred** video side-by-side  
    """)

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("""
    - **Hi, I'm Rayyan Ahmed**
    - Google Certified **AI Prompt Specialist**  
    - IBM Certified **Advanced LLM FineTuner**  
    - Hugging Face Certified: **Fundamentalist of LLMs**  
    - Expert in **EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**  
    [💼 Visit LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)
    """)

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
    - 🧠 **MediaPipe** → Face detection  
    - 🎥 **OpenCV** → Frame reading, blurring, drawing  
    - ⚙️ **NumPy** → Pixel operations  
    - 🌐 **Streamlit** → Web interface  
    """)

# ---------------- Face Blur Function ----------------
def preprocess_img(img, face_detection, top_expand=0.95, bottom_expand=0.1, side_expand=0.35, blur_ksize=(99, 99)):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)
            x1_exp = max(0, int(x1 - w * side_expand))
            y1_exp = max(0, int(y1 - h * top_expand))
            w_exp = int(w * (1 + 2 * side_expand))
            h_exp = int(h * (1 + top_expand + bottom_expand))

            # --- Blur region ---
            img[y1_exp:y1_exp + h_exp, x1_exp:x1_exp + w_exp] = cv2.blur(
                img[y1_exp:y1_exp + h_exp, x1_exp:x1_exp + w_exp], blur_ksize
            )

            # --- Green glowing border ---
            for i in range(4):
                color = (0, 255 - i * 50, 0)
                thickness = 1 + i
                cv2.rectangle(
                    img,
                    (x1_exp - i, y1_exp - i),
                    (x1_exp + w_exp + i, y1_exp + h_exp + i),
                    color,
                    thickness
                )

    return img


# ---------------- Main App ----------------
option = st.radio("🎬 Choose Input Source:", ["📁 Upload Video", "📷 Live Webcam"])

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    # ---------------- Video Upload ----------------
    if option == "📁 Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            output_path = os.path.join(tempfile.gettempdir(), "blurred_output.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            col1, col2 = st.columns(2)
            with col1:
                stframe_original = st.empty()
            with col2:
                stframe_blurred = st.empty()

            progress = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current = 0

            # 🛑 Stop button appears during processing
            # 🛑 Stop button appears before processing starts
            stop_pressed = False
            stop_button = st.button("🛑 Stop Processing", key="stop_btn", use_container_width=True)

            while True:
                if stop_button:
                    stop_pressed = True
                    st.warning("⏹️ Video processing manually stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                original_frame = frame.copy()
                blurred_frame = preprocess_img(frame, face_detection)
                out.write(blurred_frame)

                border_color = (0, 255, 0)  # Red color (BGR format)
                border_thickness = 3       # Adjust border thickness
                original_frame = cv2.copyMakeBorder(original_frame, border_thickness, border_thickness,
                                                    border_thickness, border_thickness,
                                                    cv2.BORDER_CONSTANT, value=border_color)
                blurred_frame = cv2.copyMakeBorder(blurred_frame, border_thickness, border_thickness,
                                                border_thickness, border_thickness,
                                                cv2.BORDER_CONSTANT, value=border_color)

                # --- Side by side display ---
                stframe_original.image(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB),
                                       channels="RGB", caption="🎥 Original Video")
                stframe_blurred.image(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB),
                                      channels="RGB", caption="😎 Blurred Video")

                current += 1
                progress.progress(min(current / frame_count, 1.0))

            cap.release()
            out.release()

            if not stop_pressed:
                st.success("✅ Video processing complete!")

            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download blurred video", data=f, file_name="blurred_output.mp4")

   # ---------------- Webcam Mode ----------------
    elif option == "📷 Live Webcam":
        st.info("🎦 Live webcam feed (Press Stop to end)")

        # --- Custom styled buttons ---
        st.markdown("""
            <style>
            .centered-buttons {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 20px;
            }
            div.stButton > button:first-child {
                width: 180px;
                height: 50px;
                font-size: 18px;
                border-radius: 10px;
                border: none;
                transition: all 0.3s ease;
            }
            .start-btn {
                background-color: #00FF7F;
                color: black;
                box-shadow: 0px 0px 15px #00FF7F;
            }
            .start-btn:hover {
                background-color: #00cc66;
                box-shadow: 0px 0px 25px #00FF7F;
            }
            .stop-btn {
                background-color: #FF4040;
                color: white;
                box-shadow: 0px 0px 15px #FF4040;
            }
            .stop-btn:hover {
                background-color: #cc0000;
                box-shadow: 0px 0px 25px #FF4040;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- Button layout ---
        st.markdown('<div class="centered-buttons">', unsafe_allow_html=True)
        start_btn = st.button("▶️ Start Webcam", key="start_webcam", use_container_width=False)
        stop_btn = st.button("🛑 Stop Webcam", key="stop_webcam", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        FRAME_WINDOW = st.image([])

        if start_btn:
            cap = cv2.VideoCapture(0)
            st.success("🟢 Webcam started! Processing live feed...")
            run = True
            prev_time = 0

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to access webcam.")
                    break

                # ---- FPS Calculation ----
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time

                # ---- Face Detection + Blur ----
                frame = preprocess_img(frame, face_detection)

                # ---- Green outer border ----
                border_color = (0, 255, 0)  # bright green
                border_thickness = 2       # outer glow thickness
                frame = cv2.copyMakeBorder(frame, border_thickness, border_thickness,
                                        border_thickness, border_thickness,
                                        cv2.BORDER_CONSTANT, value=border_color)

                # ---- FPS Overlay ----
                cv2.putText(frame, f"FPS: {int(fps)}", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                # ---- Display frame ----
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                if stop_btn:
                    st.warning("⏹️ Webcam stopped by user.")
                    run = False
                    break

                time.sleep(0.03)

            cap.release()
            st.success("🟢 Webcam session ended.")

      