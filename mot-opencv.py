import math
import logging
import multiprocessing
import os
import time
from queue import Empty
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import deep_sort.deep_sort.deep_sort as ds
from app import draw_rotated_box, extract_detections, putTextWithBackground

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.3
# IOU 阈值
IOU_THRESHOLD = 0.5
# 输出视频帧数
FPS = 12
# 屏幕宽度
SCREEN_WIDTH = int(2560/1.5)
# 屏幕高度
SCREEN_HEIGHT = int(1600/1.5)

logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_obj_detection_in_process(
    video_file: str,
    model_path: str,
    frame_queue: multiprocessing.Queue,
    barrier: multiprocessing.Barrier,
    stop_event: multiprocessing.Event,
    fps: int,
    track_flag: bool = True,
) -> None:
    """
    在单独的进程中运行目标检测

    Args:
        video_file (str): 视频文件路径
        model_path (str): 模型文件路径
        frame_queue (multiprocessing.Queue): 帧队列
        barrier (multiprocessing.Barrier): 同步障碍
        fps (int): 每秒帧数
        stop_event (multiprocessing.Event): 停止事件
        track_flag (bool, optional): 是否启用追踪. 默认为 True
    """
    logging.debug(f"Starting process for video {video_file}")
    model = YOLO(model_path)
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    class_names = 0

    video = cv2.VideoCapture(video_file)
    if not video.isOpened():
        logging.error(f"Error opening video file {video_file}")
        return

    try:
        original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = original_width / original_height
        logging.info(
            f"Original video dimensions: {original_width}x{original_height}, aspect ratio: {aspect_ratio:.2f}"
        )

        # 等待所有进程就绪
        barrier.wait()
        # 重置帧计数器
        fps_counter = 0
        video_fps = fps
        frame_count = 0
        start_time = time.time()

        # 读取视频帧并进行目标检测
        while not stop_event.is_set():
            ret, frame = video.read()
            if not ret:
                break

            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

            # 如果启用追踪，则更新追踪器
            if track_flag:
                detections, confarray, xyxy = extract_detections(
                    results, class_names, True
                )

                # 更新追踪器
                if len(detections) > 0:
                    results_tracker = tracker.update_obb(
                        detections, confarray, frame, xyxy
                    )
                    # 绘制追踪结果
                    for *box, track_id in results_tracker:
                        xyxyxyxy = np.array(box[:8]).reshape((4, 2)).astype(int)
                        draw_rotated_box(frame, xyxyxyxy, (255, 0, 255), 3)

                        putTextWithBackground(
                            frame,
                            str(int(track_id)),
                            (max(-10, xyxyxyxy[0][0]), max(40, xyxyxyxy[0][1])),
                            font_scale=1.5,
                            text_color=(255, 255, 255),
                            bg_color=(255, 0, 255),
                        )

            # 绘制检测结果
            res_plotted = results[0].plot(labels=False, font_size=12, line_width=2)
            # 绘制视频帧率
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps_counter = frame_count / elapsed_time

            cv2.putText(
                res_plotted,
                f"Processing FPS: {fps_counter:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                res_plotted,
                f"Video FPS: {video_fps:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            # 将处理后的帧放入队列
            frame_queue.put((
                res_plotted,
                aspect_ratio,
                original_width,
                original_height,
            ))
            # 限制帧率
            time.sleep(1 / fps)

            frame_count = 0
            start_time = time.time()

    # 视频处理完成，释放资源，发送停止信号
    finally:
        video.release()
        frame_queue.put(None)
        logging.debug(f"Stopping process for video {video_file}")


def calculate_layout(
    num_videos: int, screen_width: int, screen_height: int, aspect_ratios: List[float]
) -> List[Tuple[int, int, int, int]]:
    """
    计算视频布局

    Args:
        num_videos (int): 视频数量
        screen_width (int): 屏幕宽度
        screen_height (int): 屏幕高度
        aspect_ratios (List[float]): 视频宽高比列表

    Returns:
        List[Tuple[int, int, int, int]]: 布局列表，每个元素为 (x, y, w, h)
    """
    if num_videos == 1:
        return [(0, 0, screen_width, screen_height)]
    elif num_videos == 2:
        return [
            (0, 0, screen_width // 2, screen_height),
            (screen_width // 2, 0, screen_width // 2, screen_height),
        ]

    rows = (
        1
        if num_videos <= 2
        else (2 if num_videos <= 6 else (3 if num_videos <= 12 else 4))
    )
    cols = math.ceil(num_videos / rows)

    cell_width = screen_width // cols
    cell_height = screen_height // rows

    layout = []
    for i, aspect_ratio in enumerate(aspect_ratios):
        row = i // cols
        col = i % cols
        x = col * cell_width
        y = row * cell_height

        if aspect_ratio > cell_width / cell_height:
            width = cell_width
            height = int(width / aspect_ratio)
        else:
            height = cell_height
            width = int(height * aspect_ratio)

        width = width - (width % 2)
        height = height - (height % 2)

        x += (cell_width - width) // 2
        y += (cell_height - height) // 2

        layout.append((x, y, width, height))
        logging.debug(f"Video {i + 1} layout: x={x}, y={y}, w={width}, h={height}")

    return layout


def display_frames(
    window_name: str,
    frame_queues: List[multiprocessing.Queue],
    fps: int,
    stop_event: multiprocessing.Event,
    screen_width: int,
    screen_height: int,
) -> None:
    """显示多个视频帧

    Args:
        window_name (str): 窗口名称
        frame_queues (List[multiprocessing.Queue]): 帧队列列表
        fps (int): 每秒帧数
        stop_event (multiprocessing.Event): 停止事件
        screen_width (int): 屏幕宽度
        screen_height (int): 屏幕高度
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_width, screen_height)

    # 视频宽高比
    aspect_ratios = []
    # 视频原始尺寸
    original_sizes = []
    # 视频布局
    layout = None

    # 读取帧队列中的帧并显示
    while not stop_event.is_set():
        # 待播帧队列
        frames = []
        for i, queue in enumerate(frame_queues):
            try:
                item = queue.get(timeout=0.5)
                if item is None:
                    stop_event.set()
                    return
                frame, aspect_ratio, orig_width, orig_height = item
                frames.append(frame)
                # 保持视频宽高比和原始尺寸
                if len(aspect_ratios) < len(frame_queues):
                    aspect_ratios.append(aspect_ratio)
                    original_sizes.append((orig_width, orig_height))
                    logging.debug(
                        f"Video {i + 1} original size: {orig_width}x{orig_height}"
                    )
            except Empty:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        if not frames:
            continue
        # 计算视频布局
        if layout is None:
            layout = calculate_layout(
                len(frames), screen_width, screen_height, aspect_ratios
        )
        # 创建合并帧
        combined_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        # 调整帧大小并放入合并帧
        for i, ((x, y, width, height), frame, (orig_width, orig_height)) in enumerate(
            zip(layout, frames, original_sizes)
        ):
            # 获取当前帧大小
            current_height, current_width = frame.shape[:2]
            logging.debug(
                f"Video {i + 1} current size before resize: {current_width}x{current_height}"
            )
            # 计算缩放比例
            scale = min(width / current_width, height / current_height)
            # 调整帧大小
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)

            resized_frame = cv2.resize(frame, (new_width, new_height))
            logging.debug(f"Video {i + 1} resized to: {new_width}x{new_height}")

            # 计算偏移量
            offset_x = (width - new_width) // 2
            offset_y = (height - new_height) // 2
            # 将调整后的帧放入合并帧
            combined_frame[
                y + offset_y : y + offset_y + new_height,
                x + offset_x : x + offset_x + new_width,
            ] = resized_frame

            scale_width = new_width / orig_width
            scale_height = new_height / orig_height
            logging.debug(
                f"Video {i + 1} scale factors: width={scale_width:.2f}, height={scale_height:.2f}"
            )

        # 显示合并帧
        cv2.imshow(window_name, combined_frame)
        # 按 'q' 键退出
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    logging.debug("Display process stopped")


def main():
    # 默认视频目录和模型路径
    video_directory = "data/input"
    default_model_path = "models/yolov8-obb-v3-best.pt"

    # 支持的视频文件扩展名和视频流协议
    video_extensions = ('.mp4', '.webm')

    # 获取视频文件列表
    video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(video_extensions)]

    # 视频流 URL 支持
    video_streams = [
        # "http://example.com/stream1",
        # "rtsp://example.com/stream2"
    ]

    # 合并视频文件和视频流
    all_videos = video_files + video_streams

    # 为每个视频文件和视频流分配默认模型
    model_paths = [default_model_path for _ in all_videos]

    num_videos = len(all_videos)

    # 创建进程管理器
    manager = multiprocessing.Manager()
    # 创建同步障碍
    barrier = manager.Barrier(num_videos)
    # 创建帧队列
    frame_queues = [manager.Queue() for _ in range(num_videos)]
    # 创建停止事件
    stop_event = manager.Event()

    # 创建视频处理进程
    processes = []
    for i in range(num_videos):
        process = multiprocessing.Process(
            target=run_obj_detection_in_process,
            args=(
                video_files[i],
                model_paths[i],
                frame_queues[i],
                barrier,
                stop_event,
                FPS,
                False if i == 0 else True,
            ),
        )
        processes.append(process)
        process.start()

    # 创建显示进程
    display_process = multiprocessing.Process(
        target=display_frames,
        args=(
            "Multi-video Tracking",
            frame_queues,
            FPS,
            stop_event,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        ),
    )
    display_process.start()
    display_process.join()

    stop_event.set()
    for process in processes:
        process.join()

    logging.debug("All processes have been stopped")


if __name__ == "__main__":
    main()
