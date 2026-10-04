# 篮球检测定位项目 - 完整复现流程

## 一、环境搭建
1. 准备一台运行 Windows 或 Ubuntu 的电脑。
2. 安装 Python 3.8 或以上版本。
3. 在终端或命令行中运行以下命令，安装所需的 Python 库（推荐使用国内清华源加速）：
   pip install ultralytics opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

## 二、模型训练
1. 使用队伍提供的 YOLO 格式数据集。
2. 基于 YOLOv8n 预训练模型进行微调训练。
3. 训练参数设置为：训练轮数100轮，图像尺寸640，数据配置文件指定为 dataset/data.yaml。
4. 训练完成后，将生成的最佳模型权重保存至 runs/detect/train-5/weights/best.pt。
5. 最终模型在验证集上的 mAP50 达到了 0.913。

## 三、视频推理
1. 加载训练好的模型文件 best.pt。
2. 读取队伍提供的统一验收视频 basketball_dribble_evaluation.mp4。
3. 利用 OpenCV 逐帧读取视频画面，并对每一帧进行模型推理，识别画面中的篮球。
4. 根据推理结果，在画面中绘制绿色检测框，并附带类别标签（basketball）和置信度。

## 四、结果输出
1. 将处理后的每一帧画面重新合成视频。
2. 结果视频保存为 result_final.avi（若遇到播放器兼容问题，可使用浏览器或 PotPlayer 打开）。
3. 视频中篮球被持续标记，并能看到标签和置信度变化。

## 五、复现注意事项
1. 确保训练数据集和验收视频文件已正确放置在项目文件夹中。
2. 运行推理代码前，请检查代码中的模型路径与视频路径是否指向你电脑上的实际绝对路径。
3. 若视频文件较大，上传 GitHub 仓库时可能受限，请在 README 中说明或提供网盘链接。
