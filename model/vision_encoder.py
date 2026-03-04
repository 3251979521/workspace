import torch
from transformers import AutoProcessor, AutoModel
from transformers.image_utils import load_image
import torch.nn.functional as F
import cv2
from PIL import Image

# load the model and processor
ckpt = "google/siglip2-base-patch16-naflex"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(ckpt)
model = AutoModel.from_pretrained(ckpt).to(device).eval()

## 图片测试
# image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
# inputs = processor(images=[image], return_tensors="pt").to(device)

def extract_feature(image):
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        # 使用 get_image_features 只获取图像特征，不需要文本输入
        image_features = model.get_image_features(**inputs)
    # get_image_features 返回 BaseModelOutputWithPooling，使用 last_hidden_state
    feature = image_features.last_hidden_state.mean(dim=1).squeeze(0)
    return feature

def surprise(f1, f2):
    cos_sim = F.cosine_similarity(f1, f2, dim=0)
    return 1 - cos_sim

## 视频测试
cap = cv2.VideoCapture(r"workspace/video/1.mp4")

# 检查视频是否成功打开
if not cap.isOpened():
    print("错误：无法打开视频文件！")
    print("可能的原因：")
    print("1. 视频格式不支持")
    print("2. 缺少视频解码器")
    print("3. 文件损坏")
    exit(1)
else:
    print("视频成功打开！")
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"FPS: {fps}, 分辨率: {width}x{height}")

prev_feature = None
threshold = 0.15   # 先随便设一个

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"读取失败！已读取帧数: {frame_count}")
        break
    frame_count += 1

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    feature = extract_feature(img)

    if prev_feature is not None:
        score = surprise(feature, prev_feature)
        print("Surprise:", score.item())

        if score > threshold:
            print("=== CHUNK BOUNDARY ===")

    prev_feature = feature