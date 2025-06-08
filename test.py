import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from PIL import ImageDraw
import numpy as np

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2/sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

pil_img = Image.open("d:/projects/vision/sample.jpg").convert("RGB")
your_image = np.array(pil_img, dtype=np.uint8)

point_coords = torch.tensor([[[232.0, 70.0]]])  # shape = (1, 1, 2)
point_labels = torch.tensor([[1]])              # shape = (1, 1), 1=foreground

h, w, _ = your_image.shape
print(f"Image size: width={w}, height={h}")       # → width=954, height=349

# 4) 이미지 전체를 감싸는 박스 생성 (x_min=0, y_min=0, x_max=w-1, y_max=h-1)
#    (0 ≤ x < 954, 0 ≤ y < 349) 이므로, x_max=953, y_max=348
boxes = torch.tensor([[0.0, 0.0, w - 1.0, h - 1.0]])  # shape = (1, 4)

# with torch.inference_mode():
#     predictor.set_image(your_image)

#     masks, scores, logits = predictor.predict(
#         point_coords=point_coords,
#         point_labels=point_labels,
#         multimask_output=True   # 포인트 하나여도 후보 마스크를 3개 정도 뽑으려면 True
#     )

with torch.inference_mode():
    predictor.set_image(your_image)
    masks, scores, logits = predictor.predict(
        box=boxes,
        multimask_output=False    # 전체 이미지를 하나의 마스크로 받으려면 False
    )

# (5) masks, scores, logits 확인
# masks: torch.Tensor of shape (1, M, H', W')
# scores: torch.Tensor of shape (1, M)
# logits: torch.Tensor of shape (1, M, H', W')
print("masks.shape :", masks.shape)
print("scores.shape:", scores.shape)

mask_tensor = masks[0, 0]             # torch.Tensor (gpu/ cpu 텐서)
if isinstance(mask_tensor, np.ndarray):
    mask_np = mask_tensor             # 이미 NumPy 배열
else:
    mask_np = mask_tensor.cpu().numpy()

# 0/1 → 0~255 스케일 & uint8
mask_img = (mask_np * 255).astype(np.uint8)  # 값이 0 또는 255인 흑백 이미지

# ── 7) 흑백 마스크만 따로 저장 ────────────────────────────────────────
pil_mask = Image.fromarray(mask_img)   # mode="L"
pil_mask.save("sample2_mask_232_70.jpg")  # sample2_mask_232_70.jpg로 저장

# ── 8) 원본 이미지 위에 반투명 빨간 오버레이 합성 ───────────────────────
# (8-1) 만약 마스크 출력 해상도(h', w')가 원본 해상도(954×349)와 다르면, 리사이즈
if pil_mask.size != pil_img.size:
    pil_mask = pil_mask.resize(pil_img.size, resample=Image.NEAREST)

# (8-2) 원본 이미지를 알파 채널이 있는 RGBA로 변환
pil_img_rgba = pil_img.convert("RGBA")

# (8-3) 마스크를 빨간색 RGBA로 변환 (반투명 α=128)
mask_arr     = np.array(pil_mask)            # shape=(349, 954), 값 0 또는 255
h_resized, w_resized = mask_arr.shape

overlay_arr  = np.zeros((h_resized, w_resized, 4), dtype=np.uint8)
overlay_arr[..., 0] = 255                     # R 채널 = 255 (빨강)
overlay_arr[..., 3] = (mask_arr > 0) * 128    # mask=255인 픽셀에만 alpha=128

pil_overlay = Image.fromarray(overlay_arr, mode="RGBA")

# (8-4) 두 이미지를 알파 합성 → composited (RGBA)
composited   = Image.alpha_composite(pil_img_rgba, pil_overlay)

# (8-5) JPEG 파일로 저장하려면 RGB로 변환
composited_rgb = composited.convert("RGB")
composited_rgb.save("sample2_overlay_232_70.jpg")

print("① sample2_mask_232_70.jpg (흑백 마스크)")
print("② sample2_overlay_232_70.jpg (원본+반투명 마스크)")