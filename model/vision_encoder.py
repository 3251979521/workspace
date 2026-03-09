import torch
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt  # 引入绘图库
import numpy as np               # 用于生成刻度
from collections import deque

# load the model and processor
ckpt = "facebook/dinov2-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(ckpt)
model = AutoModel.from_pretrained(ckpt).to(device).eval()

def extract_feature(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model(**inputs)
    # 取均值降维
    feature = image_features.last_hidden_state.mean(dim=1).squeeze()
    return feature

def surprise(f1, f2):
    cos_sim = F.cosine_similarity(f1, f2, dim=0)
    return 1 - cos_sim

## 视频测试
cap = cv2.VideoCapture(r"video/1.mp4")

if not cap.isOpened():
    print("错误：无法打开视频文件！")
    exit(1)
else:
    print("视频成功打开！")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"FPS: {fps}, 分辨率: {width}x{height}")

prev_feature = None
threshold = 0.15

frame_count = 0

# 记录绘图所需的列表
frame_indices = []
surprise_scores = []

# 存储过去15帧的信息
history_features = deque(maxlen=15)
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"读取完毕或失败！总读取帧数: {frame_count}")
        break
    
    frame_count += 1

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    feature = extract_feature(img)

    if len(history_features) == 15:
        # 和半秒前（最旧的一帧）的画面进行对比
        old_feature = history_features[0]
        score = surprise(feature, old_feature)
        val = score.item()
        # 记录绘图所需的数据
        frame_indices.append(frame_count)
        surprise_scores.append(val)
        print(f"Frame {frame_count}: Surprise Score = {val:.4f}")
        # 此时的 val 会比相邻两帧比对大得多，容易触发 threshold
    
    history_features.append(feature)

cap.release()

# ========================
# 绘制 Surprise 值曲线图
# ========================
if len(frame_indices) > 0:
    plt.figure(figsize=(14, 6))
    
    # 绘制主曲线
    plt.plot(frame_indices, surprise_scores, marker='.', linestyle='-', color='b', label='Surprise Score')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

    # 标记出每一秒的帧
    # 生成 1倍fps, 2倍fps, 3倍fps... 的列表作为秒数的刻度
    max_frame = frame_count
    # 防止 fps 为 0 或 None 的情况
    if fps and fps > 0:
        second_ticks = np.arange(fps, max_frame + 1, fps) 
        
        # 用红色的虚线标注出每一秒的分界线
        for tick in second_ticks:
            plt.axvline(x=tick, color='green', linestyle=':', alpha=0.6)
            
        # 设置横坐标下方的刻度和标签，每秒显示一次（例如 1s, 2s）
        tick_labels = [f"{int(tick/fps)}s\n(F:{int(tick)})" for tick in second_ticks]
        
        # 将起点和秒数结合作为 x 轴的刻度
        all_ticks = [1] + list(second_ticks)
        all_labels = ["Start"] + tick_labels
        plt.xticks(all_ticks, all_labels)
    
    plt.xlabel('Frames / Time (Seconds)')
    plt.ylabel('Surprise Value')
    plt.title('Video Surprise Values over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存或展示图像
    plt.tight_layout()
    plt.savefig('surprise_curve.png', dpi=300)
    plt.show()
    print("曲线图绘制完成并已保存为 'surprise_curve.png'！")
else:
    print("没有提取到足够的帧进行绘图。")