# save as /root/check_parquet_rgb.py
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2

pq_path = "/data/share/250010203/data/recap558k/data/train-00000-of-00026.parquet"
out_dir = "/root/parquet_rgb_check"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_parquet(pq_path)
row = df.iloc[0]

# parquet 里 image 是 dict: {'bytes': ..., 'path': ...}
img_bytes = row["image"]["bytes"]

# PIL decode (RGB)
pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
pil_path = os.path.join(out_dir, "pil_rgb.jpg")
pil_img.save(pil_path)

# cv2 decode (BGR)
nparr = np.frombuffer(img_bytes, np.uint8)
cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cv_path = os.path.join(out_dir, "cv2_bgr.jpg")
cv2.imwrite(cv_path, cv_img)

# 打印通道均值对比
pil_np = np.array(pil_img)  # RGB
print("PIL RGB mean:", pil_np.reshape(-1, 3).mean(axis=0))  # [R, G, B]

# 如果要看 cv2 转成 RGB 后的均值
cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
print("cv2 RGB mean:", cv_rgb.reshape(-1, 3).mean(axis=0))  # [R, G, B]

print("saved:", pil_path, cv_path)
