import multiprocessing
import cv2
from ultralytics import YOLO
import math
import logging
import time
from queue import Empty
import numpy as np

import deep_sort.deep_sort.deep_sort as ds
from app import draw_rotated_box, extract_detections, putTextWithBackground


logging.basicConfig(
    filename="window_layout_debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def run_obj_detection_in_process(video_file, model_path, frame_queue, barrier, fps, stop_event, TRACK_FLAG=True):
    logging.debug(f"Starting process for video {video_file}")
    model = YOLO(model_path)
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    class_names = 0
    video = cv2.VideoCapture(video_file)
    
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height
    logging.info(f"Original video dimensions: {original_width}x{original_height}, aspect ratio: {aspect_ratio:.2f}")
    
    barrier.wait()

    
    fps_c = 0
    video_fps = fps
    frame_count = 0
    start_time = time.time()

    while not stop_event.is_set():
        ret, frame = video.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, iou=0.5)
        
        # 启用追踪
        if TRACK_FLAG:
            # 从预测结果中提取检测信息。
            detections, confarray, xyxy = extract_detections(results, class_names, True)

            if len(detections) == 0:
                continue  # 如果没有检测到任何物体，跳过当前帧
            
            resultsTracker = tracker.update_obb(detections, confarray, frame, xyxy)
            
            for *box, Id in resultsTracker:
                xyxyxyxy = np.array(box[:8]).reshape((4, 2)).astype(int)
                draw_rotated_box(frame, xyxyxyxy, (255, 0, 255), 3)

                putTextWithBackground(
                    frame,
                    str(int(Id)),
                    (max(-10, xyxyxyxy[0][0]), max(40, xyxyxyxy[0][1])),
                    font_scale=1.5,
                    text_color=(255, 255, 255),
                    bg_color=(255, 0, 255),
                )
            
        res_plotted = results[0].plot(labels=False, font_size=12, line_width=2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps_c = frame_count / elapsed_time

        cv2.putText(res_plotted, f"Processing FPS: {fps_c:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(res_plotted, f"Video FPS: {video_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_queue.put((res_plotted, aspect_ratio, original_width, original_height))

        time.sleep(1 / fps)

        frame_count = 0
        start_time = time.time()

    video.release()
    frame_queue.put(None)
    logging.debug(f"Stopping process for video {video_file}")

def calculate_layout(num_videos, screen_width, screen_height, aspect_ratios):
    if num_videos == 1:
        return [(0, 0, screen_width, screen_height)]
    elif num_videos == 2:
        return [(0, 0, screen_width // 2, screen_height), (screen_width // 2, 0, screen_width // 2, screen_height)]
    
    rows = 1 if num_videos <= 2 else (2 if num_videos <= 6 else (3 if num_videos <= 12 else 4))
    cols = math.ceil(num_videos / rows)
    
    cell_width = screen_width // cols
    cell_height = screen_height // rows
    
    layout = []
    for i, aspect_ratio in enumerate(aspect_ratios):
        row = i // cols
        col = i % cols
        x = col * cell_width
        y = row * cell_height
        
        # 计算保持宽高比的最大尺寸
        if aspect_ratio > cell_width / cell_height:
            w = cell_width
            h = int(w / aspect_ratio)
        else:
            h = cell_height
            w = int(h * aspect_ratio)
        
        # 确保宽度和高度是偶数
        w = w - (w % 2)
        h = h - (h % 2)
        
        x += (cell_width - w) // 2
        y += (cell_height - h) // 2
        
        layout.append((x, y, w, h))
        logging.debug(f"Video {i+1} layout: x={x}, y={y}, w={w}, h={h}")
    
    return layout

def display_frames(window_name, frame_queues, fps, stop_event, screen_width, screen_height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_width, screen_height)
    
    aspect_ratios = []
    frames = []
    original_sizes = []
    
    while not stop_event.is_set():
        frames = []
        for i, queue in enumerate(frame_queues):
            try:
                item = queue.get(timeout=1)
                if item is None:
                    stop_event.set()
                    return
                frame, aspect_ratio, orig_width, orig_height = item
                frames.append(frame)
                if len(aspect_ratios) < len(frame_queues):
                    aspect_ratios.append(aspect_ratio)
                    original_sizes.append((orig_width, orig_height))
                    logging.info(f"Video {i+1} original size: {orig_width}x{orig_height}")
            except Empty:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        if not frames:
            continue

        layout = calculate_layout(len(frames), screen_width, screen_height, aspect_ratios)
        combined_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        for i, ((x, y, w, h), frame, (orig_w, orig_h)) in enumerate(zip(layout, frames, original_sizes)):
            current_h, current_w = frame.shape[:2]
            logging.info(f"Video {i+1} current size before resize: {current_w}x{current_h}")
            
            # 保持宽高比的调整大小
            scale = min(w / current_w, h / current_h)
            new_w = int(current_w * scale)
            new_h = int(current_h * scale)
            
            resized_frame = cv2.resize(frame, (new_w, new_h))
            logging.info(f"Video {i+1} resized to: {new_w}x{new_h}")
            
            # 计算偏移量以居中放置视频
            offset_x = (w - new_w) // 2
            offset_y = (h - new_h) // 2
            
            combined_frame[y+offset_y:y+offset_y+new_h, x+offset_x:x+offset_x+new_w] = resized_frame
            
            scale_w = new_w / orig_w
            scale_h = new_h / orig_h
            logging.info(f"Video {i+1} scale factors: width={scale_w:.2f}, height={scale_h:.2f}")

        cv2.imshow(window_name, combined_frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    logging.debug("Display process stopped")

if __name__ == "__main__":
    screen_width = 2560
    screen_height = 1600
    video_files = [
        "Uav-Rod-9Fps.webm",
        # "Uav-Rod-9Fps.webm",
    ]
    model_paths = [
        "yolov8-obb-v3-best.engine",
        # "yolov8-obb-v3-best.pt",
    ]
    num_videos = len(video_files)
    
    manager = multiprocessing.Manager()
    barrier = manager.Barrier(num_videos)
    frame_queues = [manager.Queue() for _ in range(num_videos)]
    stop_event = manager.Event()
    fps = 12

    processes = []
    for i in range(num_videos):
        process = multiprocessing.Process(
            target=run_obj_detection_in_process,
            args=(video_files[i], model_paths[i], frame_queues[i], barrier, fps, stop_event)
        )
        processes.append(process)
        process.start()

    display_process = multiprocessing.Process(
        target=display_frames,
        args=("Multi-video Tracking", frame_queues, fps, stop_event, screen_width, screen_height)
    )
    display_process.start()

    display_process.join()

    stop_event.set()
    for process in processes:
        process.join()

    logging.debug("All processes have been stopped")