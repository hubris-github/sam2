import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from supabase import create_client, Client
import time
import glob
import uuid

supabase_url = "https://cevsjxqctilqzaeqllqc.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNldnNqeHFjdGlscXphZXFsbHFjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNDc5MjMxMSwiZXhwIjoyMDMwMzY4MzExfQ.oaEtnfGqjcMhbvNTadlOlAEf6Wji6-Qi8H2HLetOe4o"
supabase: Client = create_client(supabase_url, supabase_key)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        print(f"Mask shape: {mask.shape}")
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def update_parking_slot_status(supabase, parking_lot_id, status, slot):
    try:
        response = (
            supabase.table("parking_slot_status")
                .update({"status": status})
                .eq("parking_lot_id", parking_lot_id)
                .eq("slot_code", slot)
                .execute()
        )
        print(f"[OK] Updated {slot} to status {status}")
        return response
    except Exception as e:
        print(f"[ERROR] Failed to update {slot}: {e}")
        return None


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


sam2_checkpoint = "D:/Projects/vision/sam2/checkpoints/sam2_hiera_large.pt"
model_cfg = "D:/Projects/vision/sam2/sam2/configs/sam2/sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

def update_parking_slot_using_image():
  
  folder_path = 'D:/Projects/react/aiparking/public/uploads/'
  jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
  if not jpg_files:
    raise FileNotFoundError("해당 폴더에 jpg 파일이 없습니다.")

  latest_file = max(jpg_files, key=os.path.getmtime)
  image = Image.open(latest_file)
  # image = Image.open('D:/Projects/react/aiparking/public/uploads/parquery9.jpg')
  image = np.array(image.convert("RGB"))
  
  print("cwd:", os.getcwd(), "latest file:", latest_file)
  print("image shape:", image.shape)

  predictor.set_image(image)

  input_points = np.array([
      [210, 302],  #1
      [302, 315],  #2
      [409, 315],  #3
      [534, 296],  #4
      [658, 295],  #5
      [786, 309],  #6
      [895, 313],  #7
      [1002, 313], #8
      [1111, 343], #9
      [1214, 310], #10
      [1290, 332], #11
      [1360, 373], #12

      [134, 408],  #1
      [244, 426],  #2
      [373, 422],  #3
      [510, 426],  #4
      [663, 458],  #5
      [801, 447],  #6
      [925, 446],  #7
      [1067, 450], #8
      [1177, 423], #9
      [1264, 446], #10
      [1347, 460], #11
      [1410, 463], #12
  ])
  input_label = np.array([1])  # 양성(1)으로 설정

  print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)


  parking_lot_id = uuid.UUID("c481b6a8-9d56-49f9-a712-299f9d582d76")

  occupiedCount = 0 
  emptyCount = 0

  #                   1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24  
  lower_thresholds = [1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000,  1000 ]
  upper_thresholds = [15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000]
  score_thresholds = [0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,    0.7,  0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7,   0.7  ]
  slots = [f"A{i}" for i in range(1, 13)] + [f"B{i}" for i in range(1, 13)]

  second_point_for = {
      # 6: [397, 204],  # idx=15(16번째) --> 두 번째 좌표는 [144, 303]
      # 16: [102, 287],  # idx=16(17번째) --> 두 번째 좌표는 [ 95, 277]
      # 17: [56, 250],  # idx=16(17번째) --> 두 번째 좌표는 [ 95, 277]
  }

  for idx, pt in enumerate(input_points):
    
    if idx in second_point_for:
        # 첫 번째 좌표: input_points[15] (예: [144, 296])
        # 두 번째 좌표: 예시로 [200, 350] 을 추가 (사용자 상황에 맞춰 바꾸세요)
        first_pt = input_points[idx].tolist()  # 예: [144, 296] 또는 [ 95, 270]
        # 2) 두 번째 좌표는 미리 정의해 둔 값
        second_pt = second_point_for[idx]       # ex: [144, 303] 또는 [95, 277]

        single_point = np.array([first_pt], dtype=np.int32)  # shape = (1, 2)
        single_label = np.array([1], dtype=np.int32)         # (1,)

        # 3) 두 좌표를 합쳐서 (2,2) 모양의 배열로 만든다.
        point_pair = np.array([ first_pt, second_pt ], dtype=np.int32)
        label_pair = np.array([1, 1], dtype=np.int32)  # 둘 다 양성(1)이라고 가정

        # SAM2 predict 호출
        masks, scores, _ = predictor.predict(
            point_coords     = point_pair,
            point_labels     = label_pair,
            multimask_output = False
        )

        # 두 번째 인자로 넘긴 포인트가 두 개이므로, masks와 scores도 각각 길이 2 이상의 배열이 나옵니다.
        # 만약 multimask_output=False로 하면 masks.shape = (2, H, W) 또는 그 이상이 될 수 있으므로,
        # 보통 첫 번째 mask만 써야 한다면 masks[0], scores[0]을 그대로 사용합니다.
        mask_2d    = masks[0]    # 첫 번째 마스크
        score_val  = scores[0] 

        pixel_count = np.count_nonzero(mask_2d)
        pixel_count2 = mask_2d.sum()

        print(f"첫 번째 마스크의 픽셀 개수: {pixel_count}")   # 방법 A 결과
        print(f"첫 번째 마스크의 픽셀 개수: {pixel_count2}")  # 방법 B 결과

        lower = lower_thresholds[idx]
        upper = upper_thresholds[idx]
        # (옵션) score threshold 불러오기
        score_thresh = score_thresholds[idx]

        ys, xs = np.where(mask_2d > 0)  # or mask_2d == True

        if len(xs) == 0 or len(ys) == 0:
            width = 0
            height = 0
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            width = x_max - x_min + 1
            height = y_max - y_min + 1

        if score_val >= score_thresh and pixel_count > lower and pixel_count < upper and width < height:
            occupiedCount += 1
            print(f"idx: {idx+1}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")

            update_parking_slot_status(supabase, parking_lot_id, 1, slots[idx])

        else:
            emptyCount += 1
            print(f"idx: {idx+1}, [X] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")
            update_parking_slot_status(supabase, parking_lot_id, 0, slots[idx])

    else:
        single_point = np.array([pt])          # (1, 2)
        single_label = np.array([1])           # (1,), 양성(1)

        masks, scores, _ = predictor.predict(
            point_coords     = single_point,
            point_labels     = single_label,
            multimask_output = False
        )
        mask_2d = masks[0]
        score_val = scores[0]

        pixel_count = np.count_nonzero(mask_2d)
        pixel_count2 = mask_2d.sum()

        print(f"첫 번째 마스크의 픽셀 개수: {pixel_count}")   # 방법 A 결과
        print(f"첫 번째 마스크의 픽셀 개수: {pixel_count2}")  # 방법 B 결과

        lower = lower_thresholds[idx]
        upper = upper_thresholds[idx]
        # (옵션) score threshold 불러오기
        score_thresh = score_thresholds[idx]

        ys, xs = np.where(mask_2d > 0)  # or mask_2d == True

        if len(xs) == 0 or len(ys) == 0:
            width = 0
            height = 0
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            width = x_max - x_min + 1
            height = y_max - y_min + 1

        if score_val >= score_thresh and pixel_count > lower and pixel_count < upper and width < height:
            occupiedCount += 1
            print(f"idx: {idx+1}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")
            update_parking_slot_status(supabase, parking_lot_id, 1, slots[idx])
        else:
            emptyCount += 1
            print(f"idx: {idx+1}, [X] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")
            update_parking_slot_status(supabase, parking_lot_id, 0, slots[idx])

        
        # 화면에 띄우는 부분
        # plt.figure(figsize=(15, 15))
        # plt.imshow(image)
        # show_mask(mask_2d, plt.gca(), random_color=False, borders=True)
        # #show_points(single_point, single_label, plt.gca(), marker_size=200)
        # plt.title(f"Point #{idx+1}, Score={score_val:.3f}")
        # plt.axis('on')
        # plt.show()

  print(f"Occupied Count:  {occupiedCount}")
  print(f"Empty Count   :  {emptyCount}")

while True:
    update_parking_slot_using_image()
    time.sleep(20)  # 20초 대기