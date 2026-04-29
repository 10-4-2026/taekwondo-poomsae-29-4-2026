import streamlit as st
import cv2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from video_processor import process_video, compare_angles
import tempfile
import os
import time

# Cấu hình trang
st.set_page_config(page_title="Phân tích Tư thế Cơ thể", layout="wide")

# CSS để tối ưu di động và thẩm mỹ
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Ứng dụng Phân tích & So sánh Tư thế 🏃‍♂️")
st.markdown("Hệ thống tự động đánh giá kỹ thuật thực hiện động tác qua video.")

# Sidebar
with st.sidebar:
    st.header("Hướng dẫn")
    st.write("1. Tải lên video mẫu (chuẩn).")
    st.write("2. Tải lên video bạn thực hiện.")
    st.write("3. Nhấn 'Bắt đầu Phân tích' và đợi kết quả.")
    st.divider()
    st.info("Sử dụng MediaPipe Pose Landmarker Heavy")

# Giao diện tải video
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Video Mẫu")
    sample_file = st.file_uploader("Tải video mẫu", type=['mp4', 'mov', 'avi'], key="sample")

with col2:
    st.subheader("📁 Video Thực hiện")
    practice_file = st.file_uploader("Tải video thực hành", type=['mp4', 'mov', 'avi'], key="practice")

if sample_file and practice_file:
    if st.button("🚀 Bắt đầu Phân tích"):
        # Tạo thư mục tạm thủ công
        tmpdir = tempfile.mkdtemp()
        try:
            sample_path = os.path.join(tmpdir, "sample.mp4")
            practice_path = os.path.join(tmpdir, "practice.mp4")
            
            with open(sample_path, "wb") as f:
                f.write(sample_file.read())
            with open(practice_path, "wb") as f:
                f.write(practice_file.read())
                
            out_s_video = os.path.join(tmpdir, "res_sample.webm")
            out_p_video = os.path.join(tmpdir, "res_practice.webm")
            out_s_csv = os.path.join(tmpdir, "res_sample.csv")
            out_p_csv = os.path.join(tmpdir, "res_practice.csv")
            
            status_placeholder = st.empty()
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.write("Tiến độ Mẫu")
                p1 = st.progress(0.0)
            with p_col2:
                st.write("Tiến độ Thực hành")
                p2 = st.progress(0.0)

            try:
                # Xử lý TUẦN TỰ từng video
                status_placeholder.info("⏳ Đang xử lý lần lượt từng video...")
                
                # Xử lý video mẫu
                process_video(sample_path, out_s_video, out_s_csv, p1.progress)
                
                # Xử lý video thực hành
                process_video(practice_path, out_p_video, out_p_csv, p2.progress)
            except Exception as e:
                status_placeholder.error(f"❌ Lỗi trong quá trình xử lý video: {str(e)}")
                st.stop()

            # Đọc lại kết quả
            df_s = pd.read_csv(out_s_csv)
            df_p = pd.read_csv(out_p_csv)
            
            # Đọc dữ liệu vào bộ nhớ
            with open(out_s_video, "rb") as f:
                s_video_bytes = f.read()
            with open(out_p_video, "rb") as f:
                p_video_bytes = f.read()
            with open(out_s_csv, "rb") as f:
                s_csv_bytes = f.read()
            with open(out_p_csv, "rb") as f:
                p_csv_bytes = f.read()

            # So sánh
            metrics, df_s_res, df_p_res = compare_angles(df_s, df_p)

            # Lưu vào session_state
            st.session_state['analysis_done'] = True
            st.session_state['s_video_bytes'] = s_video_bytes
            st.session_state['p_video_bytes'] = p_video_bytes
            st.session_state['s_csv_bytes'] = s_csv_bytes
            st.session_state['p_csv_bytes'] = p_csv_bytes
            st.session_state['metrics'] = metrics
            st.session_state['df_s_res'] = df_s_res
            st.session_state['df_p_res'] = df_p_res

            time.sleep(1)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        status_placeholder.success("✅ Đã xử lý xong!")

    # Hiển thị kết quả nếu đã phân tích xong
    if st.session_state.get('analysis_done'):
        st.divider()
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("📺 Video Mẫu")
            st.video(st.session_state['s_video_bytes'])
            st.download_button("💾 Tải CSV Kết quả Mẫu", st.session_state['s_csv_bytes'], "sample_angles.csv", "text/csv")
                
        with v_col2:
            st.subheader("📺 Video Thực hiện")
            st.video(st.session_state['p_video_bytes'])
            st.download_button("💾 Tải CSV Kết quả Thực hành", st.session_state['p_csv_bytes'], "practice_angles.csv", "text/csv")
        
        st.divider()
        st.header("📊 Biểu đồ So sánh Chi tiết")
        
        metrics = st.session_state['metrics']
        df_s_res = st.session_state['df_s_res']
        df_p_res = st.session_state['df_p_res']
        
        angle_to_plot = st.selectbox("Chọn góc khớp để so sánh:", list(metrics.keys()))
        
        # Biểu đồ Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_s_res['frame'], y=df_s_res[angle_to_plot], name="Mẫu (Chuẩn)", line=dict(color='#007bff', width=3)))
        fig.add_trace(go.Scatter(x=df_p_res['frame'], y=df_p_res[angle_to_plot], name="Thực hiện", line=dict(color='#ff7f0e', width=3, dash='dash')))
        
        fig.update_layout(
            title=f"Đồ thị so sánh: {angle_to_plot}",
            xaxis_title="Thời gian (Khung hình đã đồng bộ)",
            yaxis_title="Giá trị góc (độ)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width='stretch')
        
        # Chỉ số tóm tắt
        js_val = metrics[angle_to_plot]['JS_Distance']
        avg_js = np.mean([v['JS_Distance'] for v in metrics.values()])
        
        # Quy đổi sang phần trăm tương đồng (Similarity Percentage)
        similarity_pct = (1 - avg_js) * 100

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.info(f"**Độ đo JS (góc này):** {js_val:.4f}")
        with m_col2:
            st.info(f"**Độ đo JS (trung bình):** {avg_js:.4f}")
        with m_col3:
            st.success(f"**ĐỘ TƯƠNG ĐỒNG: {similarity_pct:.1f}%**")
            
        # Bảng tổng hợp
        with st.expander("📊 Xem bảng thống kê chi tiết tất cả các khớp"):
            summary_df = pd.DataFrame([
                {"Góc Khớp": k, "Độ đo JS": round(v["JS_Distance"], 4)} 
                for k, v in metrics.items()
            ])
            st.dataframe(summary_df, width='stretch')

else:
    st.info("💡 Mẹo: Tải lên cả hai video và nhấn nút bắt đầu để xem phân tích chi tiết.")
    # Reset state if files are removed
    st.session_state['analysis_done'] = False
