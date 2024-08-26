import logging

import cv2
import gradio as gr
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from ultralytics import YOLO

import deep_sort.deep_sort.deep_sort as ds


# 控制处理流程是否终止
should_continue = True

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.2
# IOU阈值
IOU_THRESHOLD = 0.5

# 设置日志记录
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_detectable_classes(model_file: str) -> list:
    """
    获取给定模型文件可以检测的类别

    Args:
        model_file (str): 模型文件名

    Returns:
        list: 可检测的类别名称列表
    """
    model = YOLO(model_file)
    class_names = list(model.names.values())  # 获取类别名称列表
    del model  # 删除模型实例释放资源
    return class_names


def stop_processing() -> str:
    """
    终止视频处理

    Returns:
        str: 终止处理的状态消息
    """
    global should_continue
    should_continue = False  # 更改变量来停止处理
    return "尝试终止处理..."


def start_processing(
    input_path: str,
    output_path: str,
    detect_class: int,
    model: str,
    is_obb: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple:    
    """
    开始视频处理

    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出视频路径
        detect_class (int): 要检测的类别索引
        model (str): 模型文件路径
        is_obb (bool, optional): 是否使用OBB检测. 默认为 True
        progress (gr.Progress, optional): Gradio进度条对象. 默认为 gr.Progress(track_tqdm=True)

    Returns:
        tuple: 输出视频路径的元组
    """
    global should_continue
    should_continue = True

    detect_class = int(detect_class)
    # 创建YOLO模型实例
    model = YOLO(model)
    # 创建DeepSort跟踪器实例
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    
    output_video_path = detect_and_track(
        input_path, output_path, detect_class, model, tracker, is_obb
    )
    return output_video_path, output_video_path


def putTextWithBackground(
    img: np.ndarray,
    text: str,
    origin: tuple,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1,
    text_color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    thickness: int = 1,
) -> None:
    """
    绘制带有背景的文本

    Args:
        img (np.ndarray): 输入图像
        text (str): 要绘制的文本
        origin (tuple): 文本的左上角坐标
        font (int, optional): 字体类型. 默认为 cv2.FONT_HERSHEY_SIMPLEX
        font_scale (float, optional): 字体大小. 默认为 1
        text_color (tuple, optional): 文本的颜色. 默认为 (255, 255, 255)
        bg_color (tuple, optional): 背景的颜色. 默认为 (0, 0, 0)
        thickness (int, optional): 文本的线条厚度. 默认为 1
    """
    # 计算文本的尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景矩形
    bottom_left = origin
    top_right = (
        origin[0] + text_width,
        origin[1] - text_height - 5,
    )  # 减去5以留出一些边距
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # 在矩形上绘制文本
    text_origin = (origin[0], origin[1] - 5)  # 从左上角的位置减去5来留出一些边距
    cv2.putText(
        img,
        text,
        text_origin,
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def extract_detections(results: list, detect_class: int, is_obb: bool = True) -> tuple:
    """
    从模型结果中提取和处理检测信息

    Args:
        results (list): YoloV8模型预测结果
        detect_class (int): 需要提取的目标类别的索引
        is_obb (bool, optional): 是否为OBB检测结果. 默认为 True

    Returns:
        tuple: 检测结果、置信度和边界框坐标的元组
    """
    # 检测结果列表
    detections = []
    # 置信度列表
    confarray = []
    # OBB的最小外接矩形坐标列表
    xyxy_list = []

    for r in results:
        if is_obb:
            boxes = r.obb.xyxyxyxy
            xyxy = r.obb.xyxy  # 获取OBB的最小外接矩形坐标
            cls = r.obb.cls
            conf = r.obb.conf
        else:
            boxes = r.boxes.xywh  # 常规格式：x_top-left, y_top-left, width, height
            cls = r.boxes.cls
            conf = r.boxes.conf

        for i, (box, cls, conf) in enumerate(zip(boxes, cls, conf)):
            if int(cls) == detect_class:
                detections.append(box.cpu() if box.is_cuda else box)
                confarray.append(conf.cpu() if conf.is_cuda else conf)
                if is_obb:
                    xyxy_list.append(xyxy[i].cpu() if xyxy[i].is_cuda else xyxy[i])

    if detections:
        detections = np.stack(detections)
        confarray = np.stack(confarray)
        if is_obb:
            xyxy_list = np.stack(xyxy_list)
    else:
        detections = np.empty((0, 5 if is_obb else 4))
        confarray = np.empty(0)
        if is_obb:
            xyxy_list = np.empty((0, 4))

    return detections, confarray, xyxy_list if is_obb else None


def detect_and_track(
    input_path: str,
    output_path: str,
    detect_class: int,
    model: YOLO,
    tracker: ds.DeepSort,
    is_obb: bool = True,
) -> Path:
    """
    处理视频，检测并跟踪目标

    Args:
        input_path (str): 输入视频文件的路径
        output_path (str): 处理后视频保存的路径
        detect_class (int): 需要检测和跟踪的目标类别的索引
        model (YOLO): 用于目标检测的模型
        tracker (ds.DeepSort): 用于目标跟踪的模型
        is_obb (bool, optional): 是否使用OBB检测. 默认为 True

    Returns:
        Path: 处理后的视频文件路径
    """
    global should_continue

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件 {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        
        # 使用MP4V编解码器创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video_path = Path(output_path) / "output.mp4"
        output_video = cv2.VideoWriter(
            output_video_path.as_posix(), fourcc, fps, size, isColor=True
        )

        # 逐帧处理视频
        for _ in tqdm(range(total_frames)):
            if not should_continue:
                logging.info("正在停止处理")
                break

            success, frame = cap.read()
            if not success:
                logging.debug("无法读取视频帧")
                break

            results = model(
                frame, stream=True, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD
            )

            detections, confarray, xyxy = extract_detections(
                results, detect_class, is_obb
            )

            if len(detections) == 0:
                logging.debug("未检测到任何目标")
                continue

            # 更新跟踪器
            if is_obb:
                resultsTracker = tracker.update_obb(detections, confarray, frame, xyxy)
            else:
                resultsTracker = tracker.update(detections, confarray, frame)

            # 绘制检测和跟踪结果
            for *box, Id in resultsTracker:
                if is_obb:
                    xyxyxyxy = np.array(box[:8]).reshape((4, 2)).astype(int)
                    draw_rotated_box(frame, xyxyxyxy, (255, 0, 255), 3)
                else:
                    x, y, w, h = map(int, box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

                putTextWithBackground(
                    frame,
                    str(int(Id)),
                    (
                        max(-10, xyxyxyxy[0][0] if is_obb else x),
                        max(40, xyxyxyxy[0][1] if is_obb else y),
                    ),
                    font_scale=1.5,
                    text_color=(255, 255, 255),
                    bg_color=(255, 0, 255),
                )

            output_video.write(frame)

    except Exception as e:
        logging.error(f"处理视频时发生错误: {e}")
    finally:
        if "output_video" in locals():
            output_video.release()
        if "cap" in locals():
            cap.release()

    logging.info(f"输出目录为: {output_video_path}")
    return output_video_path


def draw_rotated_box(
    img: np.ndarray, xyxyxyxy: np.ndarray, color: tuple, thickness: int
) -> None:
    """
    绘制旋转的边界框

    Args:
        img (np.ndarray): 输入图像
        xyxyxyxy (np.ndarray): 旋转边界框的坐标
        color (tuple): 边界框的颜色
        thickness (int): 边界框的线条厚度
    """
    cv2.drawContours(img, [xyxyxyxy], 0, color, thickness)


def main():
    # 模型列表
    MODEL_LIST = ["models/yolov8-obb-v3-best.pt", "models/yolov8-obb-v3-best.engine"]

    # 获取模型可以检测的所有类别，默认调用MODEL_LIST中第一个模型
    detect_classes = get_detectable_classes(MODEL_LIST[0])

    # gradio界面的输入示例
    examples = [
        [
            "data/input/Uav-Rod-9Fps-30s.webm",
            "data/output",
            detect_classes[0],
            MODEL_LIST[0],
        ],
        [
            "data/input/Uav-Rod-9Fps.webm",
            "data/output",
            detect_classes[0],
            MODEL_LIST[0],
        ],
    ]

    # 使用Gradio的Blocks创建一个GUI界面
    with gr.Blocks() as demo:
        with gr.Tab("Tracking"):
            gr.Markdown(
                """
                # 无人机多目标检测与跟踪
                
                选择示例或上传自己的视频文件，选择要检测的类别和模型，然后点击"Process"按钮开始处理。
                可以使用"Stop"按钮随时终止处理，得到部分处理后的视频。
                """
            )
            with gr.Row():
                with gr.Column():
                    input_path = gr.Video(label="Input video")
                    model = gr.Dropdown(MODEL_LIST, value=MODEL_LIST[0], label="Model")
                    detect_class = gr.Dropdown(
                        detect_classes,
                        value=detect_classes[0],
                        label="Class",
                        type="index",
                    )
                    output_dir = gr.Textbox(label="Output dir", value="data/output")
                    is_obb = gr.Checkbox(label="Use OBB detection", value=True)
                    with gr.Row():
                        start_button = gr.Button("Process")
                        stop_button = gr.Button("Stop")
                with gr.Column():
                    output = gr.Video()
                    output_path = gr.Textbox(label="Output path")

                    gr.Examples(
                        examples,
                        label="Examples",
                        inputs=[input_path, output_dir, detect_class, model],
                        outputs=[output, output_path],
                        fn=start_processing,
                        cache_examples=False,
                    )

        start_button.click(
            start_processing,
            inputs=[input_path, output_dir, detect_class, model, is_obb],
            outputs=[output, output_path],
        )
        stop_button.click(stop_processing)

    demo.launch()


if __name__ == "__main__":
    main()
