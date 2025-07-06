# 이미지에서 주차 검출
#
# 수정일: 2025. 07. 06

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
#from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from supabase import create_client, Client
from datetime import datetime
import cv2
import os
from scipy.ndimage import label

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

file_template = "parquery_frame_{:05d}.jpg"
#folder_path = 'D:/Projects/vision/yolo/images/mp4/parquery_frames/'
folder_path = 'D:/Projects/vision/yolo/images/mp4/parquery_frames_0706_2/'
#folder_path = 'D:/Projects/vision/yolo/images/mp4/parquery_frames_night/'
#folder_completed_path = 'D:/Projects/vision/yolo/images/mp4/parquery1_2/'
#folder_completed_path = 'D:/Projects/vision/yolo/images/mp4/parquery1_4/'
#folder_completed_path = 'D:/Projects/vision/yolo/images/mp4/parquery_night2/'
folder_completed_path = 'D:/Projects/vision/yolo/images/mp4/parquery_0706_2_results_test/'

#start_index = 0
#end_index = 4149
#index_step = 5    # 5 seconds

start_index = 1365
end_index = 1365
index_step = 5    # 5 seconds



def overlay_mask(base_image, mask, color=(30, 144, 255), alpha=0.6):
    """
    base_image: (H, W, 3)
    mask: (H, W), dtype=bool or 0/1
    color: RGB tuple (0~255)
    alpha: blending strength (0~1)
    """
    overlay = np.zeros_like(base_image, dtype=np.uint8)
    for i in range(3):  # RGB 채널에 색상 적용
        overlay[:, :, i] = color[i]

    mask = mask.astype(bool)
    base_image[mask] = (1 - alpha) * base_image[mask] + alpha * overlay[mask]
    return base_image

def check_parking_slot_using_image():
  
  for i in range(start_index, end_index + 1, index_step):
    filename = folder_path + file_template.format(i)

    if os.path.exists(filename):
      image_pil = Image.open(filename)
      image_np = np.array(image_pil.convert("RGB"))
      
      output_image = image_np.copy()  # 마스크 누적할 이미지

      output_pil = Image.fromarray(output_image)
      draw = ImageDraw.Draw(output_pil)

      try:
            font = ImageFont.truetype("arial.ttf", 20)  # 시스템 폰트가 있으면 사용
      except:
            font = ImageFont.load_default()

      # print("cwd:", os.getcwd(), "latest file:", filename)
      # print("image shape:", image.shape)

      predictor.set_image(image_np)

      input_points = np.array([
        [206, 182],  #A1 0
        [356, 175],  #A2 1
        [528, 177],  #A3 2
        [711, 183],  #A4 3
        [882, 200],  #A5 4
        [1033, 213], #A6 5
        [1174, 242], #A7 6
        [1270, 250], #A8 7
        [1380, 270], #A9 8

        [173, 296],  #B1 9
        [289, 297],  #B2 10
        [389, 297],  #B3 11 
        [487, 326],  #B4 12 
        [644, 309],  #B5 13
        [780, 315],  #B6 14
        [903, 315],  #B7 15
        [987, 351],  #B8 16
        [1100, 340], #B9 17
        [1206, 340], #B10 18
        [1278, 356], #B11 19
        [1350, 363], #B12 20

        [124, 418],  #C1 21
        [241, 416],  #C2 22
        [377, 424],  #C3 23
        [521, 433],  #C4 24
        [664, 438],  #C5 25
        [803, 435],  #C6 26
        [942, 445],  #C7 27
        [1058, 459], #C8 28
        [1158, 470], #C9 29
        [1260, 462], #C10 30
        [1340, 468], #C11 31
        [1403, 470], #C12 32

        [101, 687],  #D1 33
        [284, 670],  #D2 34
        [559, 685],  #D3 35
        [817, 679],  #D4 36
    ])
      input_label = np.array([1])  # 양성(1)으로 설정

      # print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

      occupiedCount = 0 
      emptyCount = 0

      #                   0      1      2      3      4      5      6      7      8       9     10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36     37
      #                   1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36     37     38
      #                   A1     A2     A3     A4     A5     A6     A7     A8     A9     B1     B2     B3     B4     B5     B6     B7     B8     B9     B10    B11    B12    C1     C2     C3     C4     C5     C6     C7     C8     C9     C10    C11    C12    D1     D2     D3     D4     D5
      lower_thresholds = [3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  1000,  1000,  1000,  1000,  1000,  1000,  1000 ]
      upper_thresholds = [15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 30000, 30000, 30000, 30000, 30000]
      score_thresholds = [0.7,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6  ]
      wh_thresholds    = [0,     0,     0,     0,     0,     0,     0,     0,     0,     2,     2,     2,     222,   2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     3,     4,     5,     5,     5,     5    ]

      slots = (
            [f"A{i}" for i in range(1, 10)] +
            [f"B{i}" for i in range(1, 13)] +
            [f"C{i}" for i in range(1, 13)] +
            [f"D{i}" for i in range(1, 6)]
      )


      second_point_for = {
        2: [474, 177],   # 
        12: [538, 276],  # 
        15: [924, 290],  # 
        17: [1100, 289], # 
        26: [839, 399],  # 
      }

      overlayed_image = np.array(image_pil.convert("RGB"))

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

        # print(f"첫 번째 마스크의 픽셀 개수: {pixel_count}")   # 방법 A 결과
        # print(f"첫 번째 마스크의 픽셀 개수: {pixel_count2}")  # 방법 B 결과

        lower = lower_thresholds[idx]
        upper = upper_thresholds[idx]
        # (옵션) score threshold 불러오기
        score_thresh = score_thresholds[idx]

        mask_bool = mask_2d.astype(bool)

        # 4-연결성(상하좌우) 구조 정의
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)

        # 1) 라벨링: 각 연결된 컴포넌트에 고유 번호 부여
        labeled, num_features = label(mask_bool, structure=structure)

        # 2) 컴포넌트별 픽셀 개수 계산
        #    bincount 결과의 인덱스 i 는 라벨 번호, 값은 픽셀 수
        counts = np.bincount(labeled.ravel())

        # 3) 크기 > 30 인 컴포넌트만 마스크로 생성
        #    counts[0] 은 배경(0) 픽셀 개수이므로 제외
        large_labels = np.where(counts > 30)[0]
        large_labels = large_labels[large_labels != 0]  # 0 라벨(배경) 제거

        # 4) 최종 필터링: 크기 30 이하 컴포넌트 제거
        #    np.isin 으로 남길 라벨만 True 로
        filtered_mask_clean = np.isin(labeled, large_labels)

        # 필요 시 원래 형태(bool) 유지
        filtered_mask_clean = filtered_mask_clean.astype(bool)
        
        ys, xs = np.where(filtered_mask_clean > 0)  # or mask_2d == True

        if len(xs) == 0 or len(ys) == 0:
            width = 0
            height = 0
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            width = x_max - x_min + 1
            height = y_max - y_min + 1

        if wh_thresholds[idx] == 1: # 세로 주차
            wh_test = width < 250 and height < 250 and width < height * 2 and width < height
        elif wh_thresholds[idx] == 2:    
            wh_test = width < 250 and height < 250 and width < height * 2 and (width * 2.5 > height) and (width < height * 2.5)
        elif wh_thresholds[idx] == 3:    
            wh_test = width < 250 and height < 250 and width < height * 2 and width < height    
        elif wh_thresholds[idx] == 4:    
            wh_test = width < 250 and height < 250 and (width * 2.5 > height) and (width < height * 2.5)  
        elif wh_thresholds[idx] == 5:    
            wh_test = width < 300 and height < 250 and width < height * 2.5 and width > height    
        elif wh_thresholds[idx] == 222:    
            wh_test = width < 300 and height < 250 and width < height * 1.5 and width * 1.5 > height
        else:  # 가로 주차
            wh_test = width < 300 and height < 135 and width > height and width > height

        masked_pixels = image_np[mask_2d.astype(bool)]  # shape: (N, 3)
        unique_colors = np.unique(masked_pixels, axis=0)
        color_count = len(unique_colors)

        hsv_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        saturation_std = hsv_pixels[:, 1].std()

        aspect_ratio = width / height if height != 0 else 0
        y_center = ys.mean() if len(ys) > 0 else 0

        # if idx == 35:
        #     np.savetxt("mask_2d.csv", mask_2d, fmt="%d", delimiter=",")
        #     img = Image.fromarray((mask_2d * 255).astype(np.uint8))
        #     img.save("mask_2d.png")

        #     # mask_bool = mask_2d.astype(bool)

        #     # 4-연결성(상하좌우) 구조 정의
        #     structure = np.array([[0, 1, 0],
        #                         [1, 1, 1],
        #                         [0, 1, 0]], dtype=bool)

        #     # 1) 라벨링: 각 연결된 컴포넌트에 고유 번호 부여
        #     labeled, num_features = label(mask_bool, structure=structure)

        #     # 2) 컴포넌트별 픽셀 개수 계산
        #     #    bincount 결과의 인덱스 i 는 라벨 번호, 값은 픽셀 수
        #     counts = np.bincount(labeled.ravel())

        #     # 3) 크기 > 30 인 컴포넌트만 마스크로 생성
        #     #    counts[0] 은 배경(0) 픽셀 개수이므로 제외
        #     large_labels = np.where(counts > 30)[0]
        #     large_labels = large_labels[large_labels != 0]  # 0 라벨(배경) 제거

        #     # 4) 최종 필터링: 크기 30 이하 컴포넌트 제거
        #     #    np.isin 으로 남길 라벨만 True 로
        #     filtered_mask_clean = np.isin(labeled, large_labels)

        #     # 필요 시 원래 형태(bool) 유지
        #     filtered_mask_clean = filtered_mask_clean.astype(bool)

        #     img = Image.fromarray((filtered_mask_clean * 255).astype(np.uint8))
        #     img.save("filtered_mask.png")

        if score_val >= score_thresh and pixel_count > lower and pixel_count < upper and wh_test and saturation_std > 10 and color_count > 1500:
            occupiedCount += 1
            output_image = overlay_mask(output_image, mask_2d)
            output_pil = Image.fromarray(output_image)
            draw = ImageDraw.Draw(output_pil)  # draw 다시 초기화 필요 (PIL 객체 변경됐기 때문)

            # 텍스트 출력
            x, y = pt
            draw.text((x-30, y - 40), f"{pixel_count}", fill="yellow", font=font)
            if score_val >= 0.9:
                draw.text((x-30, y - 10), f"{score_val:.3f}", fill="yellow", font=font)
            else:
                draw.text((x-30, y - 10), f"{score_val:.2f}", fill="white", font=font)
            draw.text((x-30, y + 20), f"{saturation_std:.1f}", fill="white", font=font)  
            output_image = np.array(output_pil)

            # print(f"idx: {idx}, [O] Occupied Count: {occupiedCount}, 색상수={color_count}, 채도편차={saturation_std:.2f}, 비율={aspect_ratio:.2f}")# print(f"idx: {idx+1}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}, pixel_count={pixel_count}, saturation_std={saturation_std:.2f}, color_count={color_count}")
            # print(f"idx: {idx+1}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")
        else:
            emptyCount += 1

            if pixel_count < 50000:
                output_image = overlay_mask(output_image, mask_2d, color=(255, 144, 30), alpha=0.2)
                output_pil = Image.fromarray(output_image)
                draw = ImageDraw.Draw(output_pil)  # draw 다시 초기화 필요 (PIL 객체 변경됐기 때문)

                # 텍스트 출력
                x, y = pt
                draw.text((x-30, y - 40), f"{pixel_count}", fill="red", font=font)
                if score_val >= 0.9:
                    draw.text((x-30, y - 10), f"{score_val:.3f}", fill="red", font=font)
                else:
                    draw.text((x-30, y - 10), f"{score_val:.2f}", fill="white", font=font)
                draw.text((x-30, y + 20), f"{saturation_std:.1f}", fill="white", font=font)    
                output_image = np.array(output_pil)
            else:
                output_pil = Image.fromarray(output_image)
                draw = ImageDraw.Draw(output_pil)  # draw 다시 초기화 필요 (PIL 객체 변경됐기 때문)

                # 텍스트 출력
                x, y = pt
                draw.text((x-30, y - 40), f"{pixel_count}", fill="green", font=font)
                if score_val >= 0.9:
                    draw.text((x-30, y - 10), f"{score_val:.3f}", fill="green", font=font)
                else:
                    draw.text((x-30, y - 10), f"{score_val:.2f}", fill="green", font=font)
                draw.text((x-30, y + 20), f"{saturation_std:.1f}", fill="green", font=font)    
                output_image = np.array(output_pil)   
            # print(f"idx: {idx}, [X] Occupied Count: {occupiedCount}, 색상수={color_count}, 채도편차={saturation_std:.2f}, 비율={aspect_ratio:.2f}")
            # print(f"idx: {idx+1}, [X] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")


      file_name = os.path.basename(filename)
      name, ext = os.path.splitext(file_name)  # name = "image001", ext = ".jpg"
      file_name_with_count = f"{name}_{occupiedCount}{ext}"

      save_path = os.path.join(folder_completed_path, file_name_with_count)
    #   cv2.imwrite(save_path, overlayed_image)
      
    #   os.makedirs("output", exist_ok=True)
      Image.fromarray(output_image.astype(np.uint8)).save(save_path)
    #   print("✅ 저장 완료: output/parking_overlay.png")

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