import sys

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

# 加载 ONNX 模型
ort_session = ort.InferenceSession('./res/best.onnx')

# 预处理输入图像
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # 确保为 RGB 格式
    img = img.resize((640, 640))  # 调整大小
    img = np.array(img).astype('float32') / 255.0  # 归一化
    img = img.transpose(2, 0, 1)  # 转换为 (C, H, W)
    img = np.expand_dims(img, axis=0)  # 添加批次维度
    return img

import cv2
import torch
from PIL import Image

# Model
# model = torch.hub.load('.', 'custom', path="./res/best.pt", source='local')
# model = torch.jit.load("/Users/hfl/work/dnfm-auto/res/best.pt")
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model = YOLO('./res/best.pt')
# model.predict('./res/1.png', show=True)
model.predict(source="/Users/hfl/Desktop/dnf/f1.mp4", show=False,save=True)

# model = torch.hub.load('yolov8', 'custom', path='/Users/hfl/work/dnfm-auto/res/best.pt', source='local', force_reload=True).autoshape()

# 2. 加载和预处理图像
# img = Image.open('./res/1.png')  # 使用 PIL 打开图像
# img = img.resize((640, 640))  # 调整图像大小

# 3. 进行推理
# results=model.predict(img)
# results = model(img, size=640)  # 包含 NMS

sys.exit(0)
# Inference
# results = model(img, size=640)  # includes NMS

# Results
# results[0].print()  # print results to screen
# results[0].show()  # display results
# results[0].save()  # save as results1.jpg, results2.jpg... etc.

# Data
# print('\n', results.xyxy[0])  # print img1 predictions

# 进行推理
# input_image = preprocess_image('./res/1.png')
# outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_image})

# 处理输出
# print(outputs)


# def process_detection_output(outputs, original_image):
#     # 假设 outputs[0] 是边界框数组，形状为 (num_boxes, 6)
#     boxes = outputs[0]  # 取出边界框
#     height, width, _ = original_image.shape
#
#     for box in boxes:
#         # 确保 box 是一维数组
#         if box.ndim > 1:
#             box = box.flatten()
#
#         x1, y1, x2, y2, conf, class_id = box[:6]
#
#         # 将坐标缩放到原始图像大小
#         x1 = int(x1 * width)
#         x2 = int(x2 * width)
#         y1 = int(y1 * height)
#         y2 = int(y2 * height)
#
#         # 绘制边界框
#         cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 红色边界框
#         cv2.putText(original_image, f'Class: {int(class_id)}, Conf: {conf:.2f}',
#                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     return original_image


# 加载原始图像
# original_image = cv2.imread('./res/1.png')

# # 处理输出并绘制到原始图像上
# output_image = process_detection_output(outputs, original_image)
#
# # 显示结果
# cv2.imshow('Detected Image', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 保存结果
# cv2.imwrite('./res/output_image.jpg', output_image)
#

