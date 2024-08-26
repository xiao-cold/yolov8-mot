# 无人机多目标检测与跟踪系统

本项目是一个基于YOLOv8和DeepSORT的无人机多目标检测与跟踪系统。可以处理视频文件或实时视频流，检测和跟踪指定类别的目标。

## 功能特点

- 支持多视频同时处理和显示
- 使用YOLOv8进行目标检测
- 使用DeepSORT进行目标跟踪
- 支持常规边界框和旋转边界框(OBB)检测
- 提供Web UI界面和命令行界面
- 可调整的置信度阈值和IOU阈值
- 实时显示处理帧率和视频帧率

## 安装

1. 克隆此仓库:

```bash
git clone https://github.com/xico-cold/yolo-mot.git
cd yolo-mot
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

3. 下载旋转目标检测预训练模型并将其放在`models`目录下。

## 使用方法

### Web UI界面

1. 运行以下命令启动Web UI:

```bash
python app.py
```

2. 在浏览器中打开显示的URL。

3. 上传视频文件，选择检测类别和模型，然后点击"Process"按钮开始处理。

### OpenCV 界面

运行以下命令以使用 OpenCV 实时处理多个视频:

```bash
python mot-opencv.py
```

默认情况下，脚本会使用默认模型`models/yolov8-obb-v3-best.pt`处理`data/input`目录下的视频文件。你还可以在脚本中修改`video_streams`以处理视频流。

## 配置

你可以在`app.py`和`mot-opencv.py`文件中修改以下参数:

- `CONFIDENCE_THRESHOLD`: 检测置信度阈值
- `IOU_THRESHOLD`: IOU阈值
- `FPS`: 输出视频帧率
- `SCREEN_WIDTH`和`SCREEN_HEIGHT`: 显示窗口大小

## 开发

### 项目结构

- `app.py`: Web UI界面和单视频处理逻辑
- `mot-opencv.py`: OpenCV多视频处理逻辑
- `deep_sort/`: DeepSORT跟踪算法实现
- `models/`: 预训练模型目录
- `data/input/`: 输入视频目录
- `data/output/`: 输出视频目录


## 参考

[yolov8-deepsort-tracking](https://github.com/KdaiP/yolov8-deepsort-tracking)

## 许可证

[MIT](https://choosealicense.com/licenses/mit/)