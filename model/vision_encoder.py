import torch
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt  # 引入绘图库
import numpy as np               # 用于生成刻度
import collections

# load the model and processor
ckpt = "google/siglip2-base-patch16-naflex"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(ckpt)
model = AutoModel.from_pretrained(ckpt).to(device).eval()

def extract_feature_patch_level(image):
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # 关键修改：不要 mean()！
    # 直接返回所有 patch 的特征，形状为 [num_patches, hidden_dim]
    feature = image_features.last_hidden_state.squeeze(0) 
    return feature

def surprise_patch_level(f1, f2):
    # f1 和 f2 包含了图片各个局部的特征
    # 计算每个对应位置 patch 的余弦相似度，dim=1 表示在特征维度上计算
    cos_sim = F.cosine_similarity(f1, f2, dim=1) # 结果形状: [num_patches]
    
    # 计算每个 patch 的 surprise
    surprise_per_patch = 1 - cos_sim
    
    # 策略 1：取变化最大的那个 patch 的值作为整体的 surprise
    # max_surprise = torch.max(surprise_per_patch)
    
    # 策略 2（更稳定）：取变化最大的前 20% 的 patch 的平均值
    # 避免单个噪点引发误判
    k = max(1, int(len(surprise_per_patch) * 0.2))
    topk_surprise, _ = torch.topk(surprise_per_patch, k)
    score = torch.median(topk_surprise)
    
    return score

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

# 优化 2：扩大时间感受野（跳帧比对）
# 不要和紧挨着的前一帧比，而是和 N 帧之前（比如半秒前）的帧比。
# 这样水杯倒下的累积变化会很大，远超背景噪声。
stride = int(fps * 0.5) if fps > 0 else 15  # 设置对比跨度为 0.5 秒
feature_buffer = collections.deque(maxlen=stride)

# 优化 3：一维平滑窗口（Moving Average）
# 用来平滑最终的曲线，消除毛刺
smooth_window = 5
score_buffer = collections.deque(maxlen=smooth_window)

frame_count = 0
frame_indices = []
surprise_scores = []
smoothed_scores = [] # 用于记录平滑后的值

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    feature = extract_feature_patch_level(img)
    
    if len(feature_buffer) == stride:
        # 和 0.5 秒前的特征进行对比
        past_feature = feature_buffer[0]
        score = surprise_patch_level(feature, past_feature).item()
        
        # 将当前得分加入平滑窗口
        score_buffer.append(score)
        
        # 计算滑动平均值作为最终的 Surprise 值
        smoothed_score = np.mean(score_buffer)
        
        frame_indices.append(frame_count)
        surprise_scores.append(score) # 原始得分（可选画图）
        smoothed_scores.append(smoothed_score) # 平滑得分（主要画图）
        
        print(f"Frame {frame_count} - Smoothed Surprise: {smoothed_score:.4f}")

        if smoothed_score > threshold:
            print("=== CHUNK BOUNDARY ===")
            # 触发边界后，可以选择清空 buffer，避免同一个动作连续触发
            # feature_buffer.clear() 

    feature_buffer.append(feature)

cap.release()

# ========================
# 绘制 Surprise 值曲线图
# ========================
if len(frame_indices) > 0:
    plt.figure(figsize=(14, 6))
    
    # 绘制主曲线
    plt.plot(frame_indices, smoothed_scores, marker='.', linestyle='-', color='b', label='Surprise Score')
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