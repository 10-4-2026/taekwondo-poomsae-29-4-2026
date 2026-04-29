import cv2
import pandas as pd
import numpy as np
from pose_utils import PoseEstimator
import os
from scipy.spatial import distance

def process_video(video_path, output_video_path, output_csv_path, progress_callback=None):
    estimator = PoseEstimator()
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ưu tiên avc1 cho trình duyệt, fallback sang mp4v nếu lỗi (phổ biến trên Linux)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    all_data = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_ms = int(1000 * frame_idx / fps)
        landmarks = estimator.get_landmarks(frame, timestamp_ms)
        
        processed_frame, angles = estimator.process_frame(frame, landmarks)
        
        # Lưu dữ liệu
        data_row = {"frame": frame_idx}
        data_row.update(angles)
        all_data.append(data_row)
        
        # Hiển thị góc lên frame (tùy chọn)
        # for i, (name, val) in enumerate(angles.items()):
        #     cv2.putText(processed_frame, f"{name}: {int(val)}", (10, 30 + i*20), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(processed_frame)
        frame_idx += 1
        
        if progress_callback:
            progress = frame_idx / total_frames
            if hasattr(progress_callback, 'put'): # Nếu là multiprocessing.Queue
                progress_callback.put(progress)
            else: # Nếu là hàm thông thường
                progress_callback(progress)
        
    cap.release()
    out.release()
    estimator.close()
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False)
    return df

def compare_angles(df_sample, df_practice):
    """So sánh hai bộ dữ liệu góc"""
    # Đồng bộ hóa độ dài bằng cách nội suy (interpolation)
    common_length = max(len(df_sample), len(df_practice))
    
    def resample(df, target_len):
        x = np.linspace(0, 1, len(df))
        x_new = np.linspace(0, 1, target_len)
        new_df = pd.DataFrame({"frame": range(target_len)})
        for col in df.columns:
            if col == "frame": continue
            new_df[col] = np.interp(x_new, x, df[col])
        return new_df

    df_s_resampled = resample(df_sample, common_length)
    df_p_resampled = resample(df_practice, common_length)
    
    metrics = {}
    
    for col in df_s_resampled.columns:
        if col == "frame": continue
        
        s_vals = df_s_resampled[col].values
        p_vals = df_p_resampled[col].values
        
        # Jensen-Shannon Distance
        # Tính histogram để tạo phân phối xác suất
        h_s, _ = np.histogram(s_vals, bins=18, range=(0, 180), density=True)
        h_p, _ = np.histogram(p_vals, bins=18, range=(0, 180), density=True)
        
        # Tránh division by zero/log(0) bằng cách thêm epsilon nhỏ
        h_s += 1e-10
        h_p += 1e-10
        h_s /= h_s.sum()
        h_p /= h_p.sum()
        
        js_dist = distance.jensenshannon(h_s, h_p)
        
        metrics[col] = {
            "JS_Distance": js_dist
        }
        
    return metrics, df_s_resampled, df_p_resampled
