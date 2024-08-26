import torch
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
from tqdm.auto import tqdm

import deep_sort.deep_sort.deep_sort as ds

import gradio as gr

# 控制处理流程是否终止
should_continue = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_detectable_classes(model_file):
    """获取给定模型文件可以检测的类别。

    参数:
    - model_file: 模型文件名。

    返回:
    - class_names: 可检测的类别名称。
    """
    model = YOLO(model_file)
    class_names = list(model.names.values())  # 直接获取类别名称列表
    del model  # 删除模型实例释放资源
    return class_names


# 用于终止视频处理
def stop_processing():
    global should_continue
    should_continue = False  # 更改变量来停止处理
    return "尝试终止处理..."


# 用于开始视频处理
def start_processing(
    input_path,
    output_path,
    detect_class,
    model,
    is_obb=True,
    progress=gr.Progress(track_tqdm=True),
):
    global should_continue
    should_continue = True

    detect_class = int(detect_class)
    model = YOLO(model)
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    output_video_path = detect_and_track(
        input_path, output_path, detect_class, model, tracker, is_obb
    )
    return output_video_path, output_video_path


def putTextWithBackground(
    img,
    text,
    origin,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=1,
):
    """绘制带有背景的文本。

    :param img: 输入图像。
    :param text: 要绘制的文本。
    :param origin: 文本的左上角坐标。
    :param font: 字体类型。
    :param font_scale: 字体大小。
    :param text_color: 文本的颜色。
    :param bg_color: 背景的颜色。
    :param thickness: 文本的线条厚度。
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


def extract_detections(results, detect_class, is_obb=True):
    """
    从模型结果中提取和处理检测信息。
    - results: YoloV8模型预测结果
    - detect_class: 需要提取的目标类别的索引
    - is_obb: 是否为OBB检测结果
    """

    detections = []
    confarray = []
    xyxy_list = []

    for r in results:
        if is_obb:
            boxes = r.obb.xyxyxyxy
            xyxy = r.obb.xyxy  # 获取OBB的最小外接矩形坐标
            cls = r.obb.cls
            conf = r.obb.conf
        else:
            boxes = r.boxes.xywh  # 常规格式：x_top-left, y_top-left, width, height
            cls = r.boxs.cls
            conf = r.boxs.conf

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


# 视频处理
def detect_and_track(
    input_path: str, output_path: str, detect_class: int, model, tracker, is_obb=True
) -> Path:
    """
    处理视频，检测并跟踪目标。
    - input_path: 输入视频文件的路径。
    - output_path: 处理后视频保存的路径。
    - detect_class: 需要检测和跟踪的目标类别的索引。
    - model: 用于目标检测的模型。
    - tracker: 用于目标跟踪的模型。
    """
    global should_continue
    model = model
    cap = cv2.VideoCapture(input_path)  # 使用OpenCV打开视频文件。
    if not cap.isOpened():  # 检查视频文件是否成功打开。
        print(f"Error opening video file {input_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )  # 获取视频的分辨率（宽度和高度）。
    output_video_path = Path(output_path) / "output.mp4"  # 设置输出视频的保存路径。

    # 设置视频编码格式为XVID格式的avi文件
    # 如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    # 下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    # 下载完成后将dll文件放在当前文件夹内
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        output_video_path.as_posix(), fourcc, fps, size, isColor=True
    )  # 创建一个VideoWriter对象用于写视频。

    # 对每一帧图片进行读取和处理
    # 使用tqdm显示处理进度。
    for _ in tqdm(range(total_frames)):
        # 如果全局变量should_continue为False（通常由于GUI上按下Stop按钮），则终止目标检测和跟踪，返回已处理的视频部分
        if not should_continue:
            print("stopping process")
            break

        success, frame = cap.read()  # 逐帧读取视频。

        # 如果读取失败（或者视频已处理完毕），则跳出循环。
        if not (success):
            break

        # 使用YoloV8模型对当前帧进行目标检测。
        results = model(frame, stream=True, conf=0.2, iou=0.5)

        # 从预测结果中提取检测信息。
        detections, confarray, xyxy = extract_detections(results, detect_class, is_obb)

        if len(detections) == 0:
            continue  # 如果没有检测到任何物体，跳过当前帧

        if is_obb:
            resultsTracker = tracker.update_obb(detections, confarray, frame, xyxy)
        else:
            resultsTracker = tracker.update(detections, confarray, frame)

        for *box, Id in resultsTracker:
            if is_obb:
                # 将 box[:8] 转换为 NumPy 数组，然后调用 reshape 方法
                xyxyxyxy = np.array(box[:8]).reshape((4, 2)).astype(int)
                draw_rotated_box(frame, xyxyxyxy, (255, 0, 255), 3)
            else:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

            putTextWithBackground(
                frame,
                str(int(Id)),
                (max(-10, xyxyxyxy[0][0] if is_obb else x), max(40, xyxyxyxy[0][1] if is_obb else y)),
                font_scale=1.5,
                text_color=(255, 255, 255),
                bg_color=(255, 0, 255),
            )

        output_video.write(frame)  # 将处理后的帧写入到输出视频文件中。

    output_video.release()  # 释放VideoWriter对象。
    cap.release()  # 释放视频文件。

    print(f"output dir is: {output_video_path}")
    return output_video_path


def draw_rotated_box(img, xyxyxyxy, color, thickness):
    """绘制旋转的边界框"""
    cv2.drawContours(img, [xyxyxyxy], 0, color, thickness)


if __name__ == "__main__":
    # 模型列表，从左往右由小到大，第一次使用会自动下载
    model_list = ["yolov8-obb-v3-best.pt"]

    # 获取YoloV8模型可以检测的所有类别，默认调用model_list中第一个模型
    detect_classes = get_detectable_classes(model_list[0])

    # gradio界面的输入示例，包含一个测试视频文件路径、一个随机生成的输出目录、检测的类别、使用的模型
    examples = [
        ["Uav-Rod-9Fps-30s.webm", tempfile.mkdtemp(), detect_classes[0], model_list[0]],
    ]

    # 使用Gradio的Blocks创建一个GUI界面
    # Gradio参考文档：https://www.gradio.app/guides/blocks-and-event-listeners
    with gr.Blocks() as demo:
        with gr.Tab("Tracking"):
            # 使用Markdown显示文本信息，介绍界面的功能
            gr.Markdown(
                """
                # 目标检测与跟踪
                """
            )
            # 行容器，水平排列元素
            with gr.Row():
                # 列容器，垂直排列元素
                with gr.Column():
                    input_path = gr.Video(
                        label="Input video"
                    )  # 视频输入控件，用于上传视频文件
                    model = gr.Dropdown(
                        model_list, value=0, label="Model"
                    )  # 下拉菜单控件，用于选择模型
                    detect_class = gr.Dropdown(
                        detect_classes, value=0, label="Class", type="index"
                    )  # 下拉菜单控件，用于选择要检测的目标类别
                    output_dir = gr.Textbox(
                        label="Output dir", value=tempfile.mkdtemp()
                    )  # 文本框控件，用于指定输出视频的保存路径，默认为一个临时生成的目录
                    is_obb = gr.Checkbox(label="Use OBB detection", value=True)
                    with gr.Row():
                        # 创建两个按钮控件，分别用于开始处理和停止处理
                        start_button = gr.Button("Process")
                        stop_button = gr.Button("Stop")
                with gr.Column():
                    output = gr.Video()  # 视频显示控件，展示处理后的输出视频
                    output_path = gr.Textbox(
                        label="Output path"
                    )  # 文本框控件，用于显示输出视频的文件路径

                    # 添加示例到GUI中，允许用户选择预定义的输入进行快速测试
                    gr.Examples(
                        examples,
                        label="Examples",
                        inputs=[input_path, output_dir, detect_class, model],
                        outputs=[output, output_path],
                        fn=start_processing,  # 指定处理示例时调用的函数
                        cache_examples=False,
                    )  # 禁用示例缓存

        # 将按钮与处理函数绑定
        start_button.click(
            start_processing,
            inputs=[input_path, output_dir, detect_class, model, is_obb],
            outputs=[output, output_path],
        )
        stop_button.click(stop_processing)

    demo.launch()
