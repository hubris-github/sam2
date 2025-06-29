import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from supabase import create_client, Client
from datetime import datetime
import cv2
import os

supabase_url = "https://cevsjxqctilqzaeqllqc.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNldnNqeHFjdGlscXphZXFsbHFjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNDc5MjMxMSwiZXhwIjoyMDMwMzY4MzExfQ.oaEtnfGqjcMhbvNTadlOlAEf6Wji6-Qi8H2HLetOe4o"
supabase: Client = create_client(supabase_url, supabase_key)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#device = torch.device("cpu")
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

file_template = "frame_{:05d}.jpg"
folder_path = 'D:/Projects/vision/yolo/images/mp4/parquery_frames/'
folder_path = 'D:/Projects/vision/yolo/images/mp4/parquery_frames_night/'
folder_completed_path = 'D:/Projects/vision/yolo/images/mp4/parquery1/'
folder_completed_path = 'D:/Projects/vision/yolo/images/mp4/parquery_night/'

start_index = 120000
end_index = 739943

def check_parking_slot_using_image():
  
  for i in range(start_index, end_index + 1, 50):
    filename = folder_path + file_template.format(i)

    if os.path.exists(filename):
      image_pil = Image.open(filename)
      #image_np = np.array(image_pil.convert("RGB"))
      image_np = np.array(image_pil.convert("L"))
  
      # print("cwd:", os.getcwd(), "latest file:", filename)
      # print("image shape:", image.shape)

      gray_rgb = np.stack([image_np, image_np, image_np], axis=-1).astype(np.uint8)
      predictor.set_image(gray_rgb)

      # predictor.set_image(image_np)

      input_points = np.array([
        [206, 182],  #A1
        [356, 175],  #A2
        [528, 177],  #A3
        [711, 183],  #A4
        [882, 200],  #A5
        [1033, 213], #A6
        [1174, 242], #A7

        [173, 296],  #B1
        [289, 297],  #B2
        [389, 297],  #B3
        [518, 301],  #B4
        [644, 309],  #B5
        [780, 315],  #B6
        [903, 315],  #B7
        [1004, 328], #B8
        [1100, 340], #B9
        [1206, 340], #B10
        [1278, 356], #B11
        [1350, 363], #B12

        [124, 418],  #C1
        [241, 416],  #C2
        [377, 424],  #C3
        [521, 433],  #C4
        [664, 438],  #C5
        [803, 435],  #C6
        [942, 445],  #C7
        [1058, 459], #C8
        [1158, 470], #C9
        [1260, 462], #C10
        [1340, 468], #C11
        [1403, 470], #C12

        [101, 687],  #D1
        [284, 670],  #D2
        [559, 685],  #D3
        [817, 679],  #D4
      ])
      input_label = np.array([1])  # 양성(1)으로 설정

      # print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

      occupiedCount = 0 
      emptyCount = 0

      #                   1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36
      #                   A1     A2     A3     A4     A5     A6     A7     B1     B2     B3     B4     B5     B6     B7     B8     B9     B10    B11    B12    C1     C2     C3     C4     C5     C6     C7     C8     C9     C10    C11    C12    D1     D2     D3     D4     D5
      lower_thresholds = [3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  1000,  1000,  1000,  1000,  1000,  1000,  1000 ]
      upper_thresholds = [15000, 15000, 15000, 15000, 15000, 15000, 15000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 30000, 30000, 30000, 30000, 30000]
      score_thresholds = [0.7,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6  ]
      wh_thresholds    = [0,     0,     0,     0,     0,     0,     0,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     0,     0,     0,     0,     0    ]
      slots = (
        [f"A{i}" for i in range(1, 8)] +
        [f"B{i}" for i in range(1, 13)] +
        [f"C{i}" for i in range(1, 13)] +
        [f"D{i}" for i in range(1, 6)]
      )

      second_point_for = {
          # 6: [397, 204],  # idx=15(16번째) --> 두 번째 좌표는 [144, 303]
          # 16: [102, 287],  # idx=16(17번째) --> 두 번째 좌표는 [ 95, 277]
          # 17: [56, 250],  # idx=16(17번째) --> 두 번째 좌표는 [ 95, 277]
      }

      height, width = image_np.shape[:2]
      accumulated_mask = np.zeros((height, width), dtype=np.uint8)
      color_cycle = [(0, 0, 255),   # 빨강
               (0, 255, 0),   # 초록
               (255, 0, 0)]   # 파랑
      color_index = 0  # 색상 순환 인덱스

      # 2. 초기화
      overlayed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
      accumulated_mask = np.zeros(overlayed_image.shape[:2], dtype=np.uint8)


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

            if score_val >= score_thresh and pixel_count > lower and pixel_count < upper and width < 250 and height < 250 and width < height * 2:
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
            # mask_2d = masks[0]
            # score_val = scores[0]

            # pixel_count = np.count_nonzero(mask_2d)
            # pixel_count2 = mask_2d.sum()

            lower = lower_thresholds[idx]
            upper = upper_thresholds[idx]
            # (옵션) score threshold 불러오기
            score_thresh = score_thresholds[idx]

            for i, (mask_2d, score_val) in enumerate(zip(masks, scores)):

              # print( f"idx: {idx+1}, i={i}")

              pixel_count = np.count_nonzero(mask_2d)
              ys, xs = np.where(mask_2d > 0)

              if len(xs) == 0 or len(ys) == 0:
                  continue

              x_min, x_max = xs.min(), xs.max()
              y_min, y_max = ys.min(), ys.max()
              width = x_max - x_min + 1
              height = y_max - y_min + 1

              # print(f"width: {width}, height: {height}, score: {score_val:.3f}")

              lower = lower_thresholds[idx]
              upper = upper_thresholds[idx]
              score_thresh = score_thresholds[idx]
              wh_test = (width < 250 and height < 250 and width < height * 2) if wh_thresholds[idx] == 1 else \
                        (width < 300 and height < 135 and width > height)

              if score_val >= score_thresh and lower < pixel_count < upper and wh_test:
                  if( i == 0 ):
                      occupiedCount += 1
                      # print(f"idx: {idx+1}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}")

                  mask_bin = (mask_2d > 0).astype(np.uint8)
                  accumulated_mask = cv2.bitwise_or(accumulated_mask, mask_bin)

                  # 4. 색상 순환
                  color = color_cycle[color_index % len(color_cycle)]
                  color_index += 1

                  # 5. 색상 오버레이 생성
                  color_overlay = np.zeros_like(overlayed_image, dtype=np.uint8)
                  color_overlay[:, :] = color
                  masked_overlay = cv2.bitwise_and(color_overlay, color_overlay, mask=mask_bin)

                  # 6. 오버레이 적용
                  # alpha = 1.0
                  # overlayed_image = cv2.addWeighted(overlayed_image, 1.0, masked_overlay, alpha, 0)
                  overlayed_image[mask_bin > 0] = masked_overlay[mask_bin > 0]

      file_name = os.path.basename(filename)
      name, ext = os.path.splitext(file_name)  # name = "image001", ext = ".jpg"
      file_name_with_count = f"{name}_{occupiedCount}{ext}"

      save_path = os.path.join(folder_completed_path, file_name_with_count)
      cv2.imwrite(save_path, overlayed_image)
      
      # # BGR 변환
      # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

      # # 파란색 마스크 만들기
      # blue_overlay = np.zeros_like(image_bgr, dtype=np.uint8)
      # blue_overlay[:, :] = (0, 0, 255) # BGR에서 파란색은 (255, 0, 0)
      
      # # 마스크 적용된 파란색만 남기기
      # masked_overlay = cv2.bitwise_and(blue_overlay, blue_overlay, mask=accumulated_mask)

      # # 원본 + 마스크 혼합
      # alpha = 1.0
      # overlayed_image = cv2.addWeighted(image_bgr, 1.0, masked_overlay, alpha, 0)

      # # 경로 생성 및 저장
      # # os.makedirs(folder_completed_path, exist_ok=True)
      # file_name = os.path.basename(filename)
      # save_path = os.path.join(folder_completed_path, file_name)
      # cv2.imwrite(save_path, overlayed_image)

      print(f"Processed {filename} and saved to {save_path}")
  
check_parking_slot_using_image()

#while True:
#  update_parking_slot_using_image()
#  time.sleep(20)  # 20초 대기