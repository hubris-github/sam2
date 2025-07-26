
#
# LIVECAMERA 」草津温泉・温泉門無料駐車場
# https://www.youtube.com/watch?v=_1PfF1v9GJc
#

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
import glob
from scipy.ndimage import label
import skimage.measure
from typing import List, Tuple

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

def count_edges_in_roi(
    image_np: np.ndarray,
    mask_2d: np.ndarray,
    x_min: int,
    y_min: int,
    width: int,
    height: int,
    low_thresh: int = 50,
    high_thresh: int = 150,
    blur_ksize: tuple = (5, 5)
) -> int:
    """
    mask_2d로 정의된 영역 안에서 Canny 에지를 검출하고 에지 픽셀 수를 반환합니다.

    Args:
        image_np: (H, W, 3) RGB 이미지
        mask_2d:  (H, W) bool 또는 0/1 바이너리 마스크
        x_min:    ROI 왼쪽 상단 X 좌표
        y_min:    ROI 왼쪽 상단 Y 좌표
        width:    ROI 너비
        height:   ROI 높이
        low_thresh:  Canny 하한값
        high_thresh: Canny 상한값
        blur_ksize:  GaussianBlur 커널 크기

    Returns:
        edge_count: ROI 내에서 검출된 에지 픽셀(255) 개수
    """
    
    H, W = mask_2d.shape
    if width <= 0 or height <= 0:
        return 0
    x0, y0 = x_min, y_min
    x1, y1 = x_min + width, y_min + height
    # 영상 범위 밖이면 잘라 맞추기
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x0 >= x1 or y0 >= y1:
        return 0
    
    # 1) 그레이스케일 변환
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 2) ROI & 마스크 영역 잘라내기
    roi_gray = gray[y_min:y_min+height, x_min:x_min+width]
    mask_roi = (mask_2d[y_min:y_min+height, x_min:x_min+width]
                .astype(np.uint8) * 255)

    # 4) 마스크된 픽셀 체크
    if mask_roi.sum() == 0:
        return 0
    
    # 3) 마스크 적용
    roi_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=mask_roi)    

    # 4) 노이즈 제거
    roi_blur = cv2.GaussianBlur(roi_masked, blur_ksize, 0)

    # 5) Canny 에지 검출
    edges_roi = cv2.Canny(roi_blur, low_thresh, high_thresh)

    # 6) 에지 픽셀 수 세기
    edge_count = int(np.count_nonzero(edges_roi))
    return edge_count

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


def get_overlap_pixels(rect_points, mask_2d: np.ndarray) -> int:
    """
    다각형(rect_points)과 이진 마스크(mask_uint8)의 겹치는 픽셀 수를 계산해서 반환합니다.

    Parameters:
    -----------
    rect_points : array-like of shape (N, 2)
        다각형을 정의하는 (x, y) 좌표 리스트 또는 배열.
    mask_uint8 : numpy.ndarray, dtype=uint8
        0 또는 1 값으로 이루어진 2D 이진 마스크.

    Returns:
    --------
    int
        다각형 영역과 마스크가 겹치는 픽셀의 개수.
    """

    if mask_2d.dtype != np.uint8:
        # mask_2d가 float32라면 0보다 큰 픽셀을 1로, 나머지를 0으로
        mask_uint8 = (mask_2d > 0).astype(np.uint8)
    else:
        mask_uint8 = mask_2d

    # 1) 좌표를 int32 numpy 배열로 변환
    pts = np.array(rect_points, dtype=np.int32)

    # 2) mask와 동일 크기의 빈 마스크 생성
    poly_mask = np.zeros_like(mask_uint8, dtype=np.uint8)

    # 3) 다각형 내부를 1로 채우기
    cv2.fillPoly(poly_mask, [pts], 1)

    # 4) AND 연산하여 겹치는 부분만 남기고, 픽셀 수 세기
    intersection = mask_uint8  & poly_mask
    return int(np.count_nonzero(intersection))

def remove_small_components(mask_2d: np.ndarray,
                            min_size: int = 500,
                            connectivity: np.ndarray = None) -> np.ndarray:
    
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
    large_labels = np.where(counts > min_size)[0]
    large_labels = large_labels[large_labels != 0]  # 0 라벨(배경) 제거

    # 4) 최종 필터링: 크기 30 이하 컴포넌트 제거
    #    np.isin 으로 남길 라벨만 True 로
    filtered_mask_clean = np.isin(labeled, large_labels)

    # 필요 시 원래 형태(bool) 유지
    filtered_mask_clean = filtered_mask_clean.astype(bool)

    # 방법 1: np.where 로 새 배열 만들기
    mask_2d = np.where(filtered_mask_clean, mask_2d, 0)
    
    return mask_2d


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


sam2_checkpoint = "D:/Projects/vision/sam2/checkpoints/sam2_hiera_large.pt"
model_cfg = "D:/Projects/vision/sam2/sam2/configs/sam2/sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

#folder_path = 'D:/Projects/vision/yolo/images/mp4/japan/'
folder_path = 'D:/Projects/vision/capture_images/20250714/'
folder_completed_path = 'D:/Projects/vision/capture_images/20250714/parquery2/'

start_index = 0 #0
end_index = 0
index_step = 1

# def overlay_mask(base_image, mask, color=(30, 144, 255), alpha=0.6):
#     """
#     base_image: (H, W, 3)
#     mask: (H, W), dtype=bool or 0/1
#     color: RGB tuple (0~255)
#     alpha: blending strength (0~1)
#     """
#     overlay = np.zeros_like(base_image, dtype=np.uint8)
#     for i in range(3):  # RGB 채널에 색상 적용
#         overlay[:, :, i] = color[i]

#     mask = mask.astype(bool)
#     base_image[mask] = (1 - alpha) * base_image[mask] + alpha * overlay[mask]
#     return base_image

def overlay_mask(base_image, mask, color=(30, 144, 255), alpha=0.5):
    """
    base_image: (H, W, 3)  np.ndarray (BGR)
    mask:       (H, W)     np.ndarray, dtype=bool or 0/1
    color:      RGB tuple (0~255)
    alpha:      blending strength (0~1)
    """
    # ① base_image를 3채널 BGR로 맞추기 (혹시 그레이이면 복사+변환)
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        base_image = base_image.copy()

    h, w = base_image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        overlay[:, :, i] = color[i]

    mask = mask.astype(bool)
    # ② Boolean mask 영역만 blending
    base_image[mask] = (
        (1 - alpha) * base_image[mask] +
         alpha      * overlay[mask]
    ).astype(np.uint8)

    return base_image

def write_log(message: str) -> None:
    log_filename = folder_completed_path + "processing.log"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"{timestamp} [{message}]\n"
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(log_line)

# rectangles_as_tuples 은 List[List[Tuple[int,int]]] 형태
# 각 inner list 는 네 개의 (x, y) 꼭짓점 좌표를 담고 있다고 가정합니다.
def is_within_rectangle(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    rect_pts: List[Tuple[int,int]]
) -> bool:
    """
    rect_pts: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    반환: rect_pts 가 정의하는 사각형 내부에
         [x_min, x_max] × [y_min, y_max] 영역이
         완전히 포함되면 True, 아니면 False
    """
    # 1) “값이 하나라도 0” 조건
    if any(x == 0 or y == 0 for x, y in rect_pts):
        return True

    # 2) 실제 포함 여부 계산
    xs = [p[0] for p in rect_pts]
    ys = [p[1] for p in rect_pts]
    rect_x_min, rect_x_max = min(xs), max(xs)
    rect_y_min, rect_y_max = min(ys), max(ys)

    return (
        rect_x_min <= x_min and
        x_max      <= rect_x_max and
        rect_y_min <= y_min and
        y_max      <= rect_y_max
    )

def check_parking_slot_using_image():
  
    pattern = folder_path + "4_*sshot.png"
    files = glob.glob(pattern)

    for index, filename in enumerate(files[start_index : end_index + 1 : index_step]):  

        filename = folder_path + "4_20250714_120749sshot.png"

        print(f"{start_index+index}, Processing file: {filename}")

        if os.path.exists(filename):
            image_pil = Image.open(filename)
            image_np = np.array(image_pil.convert("RGB"))
            # image_np = np.array(image_pil.convert("L"))

            gray = np.array(image_pil.convert("L"))  # Convert to grayscale

            # 3) CLAHE로 대비 향상
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # 4) 언샤프 마스크로 샤프닝
            blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3, sigmaY=3)
            sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

            # 5) 캐니 에지 검출
            edges = cv2.Canny(sharpened, threshold1=50, threshold2=150)

            # 6) 에지 팽창
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)

            # 7) BGR 변환 및 빨간색 에지 오버레이
            bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            overlay = bgr.copy()
            overlay[edges_dilated > 0] = (0, 0, 255)

            # 8) 원본과 합성
            output = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
            
            mask_color = np.zeros_like(image_np)
            # edges_dilated > 0 인 픽셀에 빨간(RGB) 채널만 255
            mask_color[edges_dilated > 0] = [255, 0, 0]

            # 2) 원본과 블렌딩
            alpha = 0.7  # 원본 가중치
            beta  = 0.3  # 마스크 가중치
            # output_image = (image_np * alpha + mask_color * beta).astype(np.uint8)

            output_image = cv2.addWeighted(image_np, alpha, mask_color, beta, 0)

            # output_image = image_np.copy()  # 마스크 누적할 이미지


            try:
                    font = ImageFont.truetype("arial.ttf", 15)
                    font30 = ImageFont.truetype("arial.ttf", 30)
                    font100 = ImageFont.truetype("arial.ttf", 100)
                    font_edge = ImageFont.truetype("arial.ttf", 20)
            except:
                    font = ImageFont.load_default()
                    font30 = ImageFont.truetype("arial.ttf", 30)
                    font100 = ImageFont.load_default()
                    font_edge = ImageFont.truetype("arial.ttf", 20)

            # print("cwd:", os.getcwd(), "latest file:", filename)
            # print("image shape:", image.shape)

            predictor.set_image(image_np)
            # gray_rgb = np.stack([image_np, image_np, image_np], axis=-1).astype(np.uint8)
            # predictor.set_image(gray_rgb)

            input_points = np.array([
                [144, 152],  #A1 0
                [259, 149],  #A2 1
                [392, 148],  #A3 2
                [531, 149],  #A4 3
                [658, 160],  #A5 4
                [792, 171], #A6 5
                [895, 185], #A7 6
                [987, 199], #A8 7
                [1067, 217], #A9 8

                [142, 256],  #B1 9
                [232, 258],  #B2 10
                [319, 255],  #B3 11 
                [416, 259],  #B4 12 
                [515, 263],  #B5 13
                [613, 266],  #B6 14
                [703, 270],  #B7 15
                [780, 278],  #B8 16
                [867, 278],  #B9 17
                [944, 287],  #B10 18
                [1014, 296], #B11 19
                [1072, 302], #B12 20

                [81, 354],  #C1 21
                [187, 351], #C2 22
                [292, 355], #C3 23
                [401, 357], #C4 24
                [517, 362], #C5 25
                [623, 363], #C6 26
                [732, 369], #C7 27
                [828, 372], #C8 28
                [918, 374], #C9 29
                [993, 383], #C10 30
                [1064, 386], #C11 31
                [1112, 394], #C12 32

                [24, 546],   #D1 33
                [168, 552],  #D2 34
                [403, 560],  #D3 35
                [631, 565],  #D4 36
                [910, 543],  #D4 37
            ])
            
            input_label = np.array([1])  # 양성(1)으로 설정

            # print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

            occupiedCount = 0 
            emptyCount = 0

            #                   0      1      2      3      4      5      6      7      8       9     10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36     37
            #                   1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36     37     38
            #                   A1     A2     A3     A4     A5     A6     A7     A8     A9     B1     B2     B3     B4     B5     B6     B7     B8     B9     B10    B11    B12    C1     C2     C3     C4     C5     C6     C7     C8     C9     C10    C11    C12    D1     D2     D3     D4     D5
            lower_thresholds = [2000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  1000,  1000,  1000,  1000,  1000,  1000,  1000 ]
            upper_thresholds = [15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 30000, 30000, 30000, 30000, 30000]
            score_thresholds = [0.7,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.3,   0.25,  0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6  ]
            wh_thresholds    = [1,     1,     1,     2,     2,     2,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     2,     2,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1    ]
            vehicleDetected  = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
            width_thresholds = [120,   150,   150,   150,   150,   150,   130,   130,   130,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   170,   170,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   230,   230,   240,   230,   230    ]
            height_thresholds= [70,    80,    80,    80,    80,    80,    80,    80,    130,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   150,   120,   120,   120,   120,   120    ]
            wh_direction     = [-1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    0,     0,     0,     1,     1,     1,     1,     1,     1,     1,     -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    0,     0,     0,     1,     1,     1,     1,     1,     1,     1,     1,     1    ]
            

            rectangles_as_tuples = [
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 1 (0)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 2 (1)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 3 (2)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 4 (3)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 5 (4)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 6 (5)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 7 (6)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 8 (7)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 9 (8)

                [(0, 0), (0, 0), (0, 0), (0, 0)], # 10 (9)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 11 (10)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 12 (11)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 13 (12)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 14 (13)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 15 (14)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 16 (15)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 17 (16)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 18 (17)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 19 (18)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 20 (19)
                [(1041, 234), (1041, 337), (1120, 337), (1120, 234)], # 21 (20)

                [(0, 0), (0, 0), (0, 0), (0, 0)], # 22 (21)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 23 (22)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 24 (23)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 25 (24)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 26 (25)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 27 (26) 
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 28 (27)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 29 (28)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 30 (29)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 31 (30) 
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 32 (31)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 33 (32)

                [(0, 0), (0, 0), (0, 0), (0, 0)], # 34 (33)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 35 (34)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 36 (35)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 37 (36)
                [(0, 0), (0, 0), (0, 0), (0, 0)], # 38 (37)
            ]

            upanddown_as_tuples = [
                [(152, 106), (152, 126), (200, 126), (200, 106)], #1 (0)
                [(270, 106), (270, 126), (310, 126), (310, 106)], #2 (1)
                [(400, 106), (400, 126), (460, 126), (460, 106)], #3 (2)                
                [(539, 106), (539, 126), (592, 126), (592, 106)], #4 (3)
                [(667, 132), (667, 152), (720, 132), (720, 152)], #5 (4)
                [(794, 123), (794, 143), (843, 152), (843, 123)], #6 (5)
                [(894, 141), (894, 161), (940, 163), (940, 141)], #7 (6)
                [(991, 166), (991, 186), (1032, 183), (1032, 166)], #8 (7)
                [(1082, 180), (1082, 200), (1114, 200), (1114, 180)], #9 (8)

                [(140, 200), (140, 215), (184, 215), (184, 200)], #10 (9)
                [(243, 199), (243, 212), (287, 212), (287, 199)], #11 (10)
                [(324, 199), (324, 213), (364, 213), (364, 199)], #12 (11)
                [(420, 199), (420, 213), (469, 213), (469, 199)], #13 (12)
                [(520, 199), (520, 215), (572, 215), (572, 199)], #14 (13)
                [(616, 207), (616, 219), (667, 219), (667, 207)], #15 (14)
                [(707, 212), (707, 224), (752, 224), (752, 212)], #16 (15)
                [(793, 216), (793, 230), (847, 230), (847, 216)], #17 (16)
                [(874, 222), (874, 236), (921, 236), (921, 222)], #18 (17)                
                [(941, 232), (941, 246), (982, 246), (982, 232)], # 19 (18)
                [(1008, 235), (1008, 253), (1038, 253), (1038, 235)], # 20 (19)
                [(1061, 242), (1061, 260), (1093, 260), (1093, 242)], # 21 (20)

                [(98, 283), (98, 301), (142, 298), (142, 283)], # 22 (21)
                [(200, 283), (200, 301), (252, 298), (252, 283)], # 23 (22)
                [(310, 280), (310, 301), (378, 301), (378, 280)], # 24 (23)
                [(424, 290), (424, 302), (470, 302), (470, 290)], # 25 (24)
                [(530, 292), (530, 306), (582, 306), (582, 292)], # 26 (25)
                [(637, 292), (637, 308), (687, 308), (687, 292)], # 27 (26) 
                [(732, 297), (732, 309), (782, 309), (782, 297)], # 28 (27)
                [(821, 305), (821, 318), (877, 318), (877, 305)], # 29 (28)
                [(909, 311), (909, 324), (960, 324), (960, 311)], # 30 (29)
                [(980, 314), (980, 334), (1027, 334), (1027, 314)], # 31 (30) 
                [(1050, 322), (1050, 329), (1087, 329), (1087, 322)], # 32 (31)
                [(1104, 326), (1104, 335), (1130, 335), (1130, 326)], # 33 (32)

                [(12, 493), (12, 510), (73, 510), (73, 493)], # 34 (33)
                [(186, 499), (186, 520), (278, 520), (278, 499)], # 35 (34)
                [(378, 501), (378, 529), (516, 529), (516, 501)], # 36 (35)
                [(601, 505), (601, 528), (738, 528), (738, 505)], # 37 (36)
                [(854, 502), (854, 519), (991, 519), (991, 502)], # 38 (37)
            ]

            # idx 18 ~ 37(여기 18면에는 왼쪽 면이 없으므로 임시로 왼쪽면을 만들어 줌.)
            # idx 0 ~ 17 
            rectangles_as_tuples2 = [
                [(1352, 569), (1419, 657), (1429, 651), (1368, 568)], # idx 15(#16)
                [(1404, 556), (1427, 573), (1427, 558), (1422, 548)], # idx 17(#18)
                [(165, 405), (114, 446), (129, 452), (188, 406)], # idx 18(#19)
            ]

            three_point_for = [
                [   # #1 그룹
                    [(502, 618), (502, 626), (510, 626), (510, 618)],
                    [(538, 612), (538, 622), (559, 622), (559, 612)],
                    [(576, 623), (576, 634), (587, 634), (587, 623)]
                ],
                [   # #2 그룹
                    [(0, 0), (0, 0), (0, 0), (0, 0)],
                    [(0, 0), (0, 0), (0, 0), (0, 0)],
                    [(0, 0), (0, 0), (0, 0), (0, 0)],
                ]
            ]

            second_point_for = {
                0: [198, 147],
                1: [331, 148],
                2: [458, 149],
                3: [589, 155],
                4: [720, 164],
                5: [839, 176],
                6: [938, 192],
                7: [1024, 203],
                8: [1103, 225],

                9: [185, 228],
               10: [275, 225],
               11: [366, 224],
               12: [460, 226],
               13: [554, 229],
               14: [649, 232],
               15: [726, 239],
               16: [797, 243],
               17: [877, 250],
               18: [956, 259],
               19: [1021, 266],
               20: [1074, 275],

               21: [147, 314],
               22: [239, 304],
               23: [344, 309],
               24: [454, 315],
               25: [561, 316],
               26: [655, 320],
               27: [763, 325],
               28: [849, 332],
               29: [933, 334],
               30: [1002, 339],
               31: [1064, 350],
               32: [1116, 353],

               33: [84, 546],
               34: [263, 564],
               35: [480, 560],
               36: [731, 567],
               37: [976, 536],
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
                
                # print("mask_2d shape:", mask_2d.shape, " dtype:", mask_2d.dtype)

                # props = skimage.measure.regionprops(mask_2d.astype(int))
                # ar = props[0].minor_axis_length / props[0].major_axis_length
                # solidity = props[0].solidity

                # print(f"idx: {idx}, solidity: {solidity:.3f}")

                # 필터 영역 출력
                # selected_idx = 16
                # if idx == selected_idx:
                #     # np.savetxt("mask_2d.csv", mask_2d, fmt="%d", delimiter=",")
                #     img = Image.fromarray((mask_2d * 255).astype(np.uint8))
                #     img.save("mask_2d.png")

                #     mask_bool = mask_2d.astype(bool)

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
                #     large_labels = np.where(counts > 500)[0]
                #     large_labels = large_labels[large_labels != 0]  # 0 라벨(배경) 제거

                #     # 4) 최종 필터링: 크기 30 이하 컴포넌트 제거
                #     #    np.isin 으로 남길 라벨만 True 로
                #     filtered_mask_clean = np.isin(labeled, large_labels)

                #     # 필요 시 원래 형태(bool) 유지
                #     filtered_mask_clean = filtered_mask_clean.astype(bool)

                #     # 방법 1: np.where 로 새 배열 만들기
                #     mask_2d = np.where(filtered_mask_clean, mask_2d, 0)

                #     # print("#mask_2d shape:", mask_2d.shape, " dtype:", mask_2d.dtype)
                #     # print("filtered mask_2d shape:", mask_2d.shape, "nonzero count:", np.count_nonzero(mask_2d))

                #     # 방법 2: in-place 인덱싱 (원본 배열 변경)
                #     # mask_2d = mask_2d.copy()            # 혹시 원본을 보존하고 싶다면
                #     # mask_2d[~filtered_mask_clean] = 0

                #     img = Image.fromarray((filtered_mask_clean * 255).astype(np.uint8))
                #     img.save("mask_2d_filtered.png")

                # 500점 이하 노이즈 제거
                mask_2d = remove_small_components(mask_2d, min_size=500)

                ##########################################################
                #
                #  검출된 영역이 왼쪽이나 오른쪽 주차면을 침범하는지 여부 확인  ##
                #
                ##########################################################

                isPixelOverlappingUp = True
                isPixelOverlappingRight = True
                isPixelOverlappingLeft = True
                isPixelOverlappingSpace = True
                isSpecialCase = False

                up_overlap_pixels = get_overlap_pixels(upanddown_as_tuples[idx], mask_2d)
                write_log(f"idx+1: {idx+1}, (U) up_overlap_pixels = {up_overlap_pixels}")

                if up_overlap_pixels == 0:  # 왼쪽 주차면을 침범해야 함, 침범하지 않으면 주차된 차량이 없는 경우임
                    isPixelOverlappingUp = False

                pixel_count = np.count_nonzero(mask_2d)
                pixel_count2 = mask_2d.sum()

                # print(f"첫 번째 마스크의 픽셀 개수: {pixel_count}")   # 방법 A 결과
                # print(f"첫 번째 마스크의 픽셀 개수: {pixel_count2}")  # 방법 B 결과

                lower = lower_thresholds[idx]
                upper = upper_thresholds[idx]
                # (옵션) score threshold 불러오기
                score_thresh = score_thresholds[idx]

                mask_bool = mask_2d.astype(bool)

                # SAM2 버그로 검출된 영역과 별도로 점들이 생기는 문제가 있음(BEGIN)
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
                large_labels = np.where(counts > 0)[0]
                large_labels = large_labels[large_labels != 0]  # 0 라벨(배경) 제거

                # 4) 최종 필터링: 크기 30 이하 컴포넌트 제거
                #    np.isin 으로 남길 라벨만 True 로
                filtered_mask_clean = np.isin(labeled, large_labels)

                # 필요 시 원래 형태(bool) 유지
                filtered_mask_clean = filtered_mask_clean.astype(bool)
                # SAM2 버그로 검출된 영역과 별도로 점들이 생기는 문제가 있음(END)
                
                # ys, xs = np.where(filtered_mask_clean > 0)  # or mask_2d == True
                ys, xs = np.where(mask_2d > 0)  # or mask_2d == True

                if len(xs) == 0 or len(ys) == 0:
                    width = 0   
                    height = 0
                else:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    width = x_max - x_min + 1
                    height = y_max - y_min + 1

                wh_test = width <= width_thresholds[idx] and height <= height_thresholds[idx] 

                # (5) mask_2d를 bool로 변환한 뒤, filtered_mask_clean과 AND 연산
                mask_bool = mask_2d.astype(bool) & filtered_mask_clean

                # bool 마스크를 이용해 원본 이미지에서 픽셀 추출
                masked_pixels = image_np[mask_bool]  # shape: (N, 3)

                # masked_pixels = image_np[mask_2d.astype(bool)]  # shape: (N, 3)
                unique_colors = np.unique(masked_pixels, axis=0)
                color_count = len(unique_colors)

               
                edge_cnt = count_edges_in_roi(image_np, mask_2d, x_min, y_min, width, height)

                isWithInRectangle = is_within_rectangle(x_min, y_min, x_max, y_max, rectangles_as_tuples[idx])

                write_log(f"=" * 120)
                write_log(f"idx+1: {idx+1}, pixel_count               = {pixel_count}")
                write_log(f"idx+1: {idx+1}, score_val >= score_thresh = {score_val >= score_thresh}, {score_val}, {score_thresh}")
                write_log(f"idx+1: {idx+1}, pixel_count > lower       = {pixel_count > lower}, {pixel_count}, {lower}")
                write_log(f"idx+1: {idx+1}, pixel_count < upper       = {pixel_count < upper}, {pixel_count}, {upper}")
                write_log(f"idx+1: {idx+1}, isPixelOverlapping(LRUPS) = {isPixelOverlappingLeft}, {isPixelOverlappingRight}, {isPixelOverlappingUp}, {isPixelOverlappingSpace}, {isSpecialCase}")
                write_log(f"idx+1: {idx+1}, isWithInRectangle         = {isWithInRectangle}")
                write_log(f"idx+1: {idx+1}, wh_test                   = {wh_test}, width={width}, height={height}")
                write_log(f"idx+1: {idx+1}, x, y, xmax, ymax          = {x_min}, {y_min}, {x_max}, {y_max}")
                write_log(f"idx+1: {idx+1}, colors, edge_cnt          = {color_count}, {edge_cnt}")
                write_log(f"idx+1: {idx+1}, pt                        = {pt}")

                isCondition1 = score_val >= score_thresh and pixel_count > lower and pixel_count < upper and isWithInRectangle
                isCondition2 = isPixelOverlappingUp and isPixelOverlappingRight and isPixelOverlappingSpace
                
                if isCondition1 and isCondition2 and ((isPixelOverlappingLeft and wh_test) or isSpecialCase):

                    vehicleDetected[idx] = True
                    occupiedCount += 1

                    write_log(f"-" * 120)
                    write_log(f"idx+1: {idx+1}, [# OOO #] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, Score={score_val:.3f}, width={width}, height={height}")
                    write_log(f"=-" * 60)
                    write_log(f" " * 10)
                    write_log(f" " * 10)

                    output_image = overlay_mask(output_image, mask_2d)
                    output_pil = Image.fromarray(output_image)
                    draw = ImageDraw.Draw(output_pil)  # draw 다시 초기화 필요 (PIL 객체 변경됐기 때문)

                    # 텍스트 출력
                    x, y = pt
                    draw.text((x-10, y - 70), f"T", fill="yellow", font=font30)
                    draw.text((x-10, y - 40), f"{idx}", fill="green", font=font)
                    draw.text((x-10, y - 25), f"{edge_cnt}", fill="yellow", font=font_edge)
                    draw.text((x-25, y + 10), f"{width:.0f},{height:.0f}", fill="red", font=font)   

                    if idx in [21, 22]:
                        xs = [p[0] for p in upanddown_as_tuples[idx]]
                        ys = [p[1] for p in upanddown_as_tuples[idx]]
                        rect_x_min, rect_x_max = min(xs), max(xs)
                        rect_y_min, rect_y_max = min(ys), max(ys)
                        draw.rectangle(
                            [(rect_x_min, rect_y_min), (rect_x_max, rect_y_max)],
                            outline="yellow", width=1
                        )

                    output_image = np.array(output_pil)

                else:
                    write_log(f"-" * 120)
                    write_log(f"idx+1: {idx+1}, [# XXX #] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, Score={score_val:.3f}, width={width}, height={height}, 색상수={color_count}")
                    write_log(f"=-" * 60)
                    write_log(f" " * 10)
                    write_log(f" " * 10)

                    emptyCount += 1

                    if pixel_count < 30000:
                        
                        if idx in [31]:
                            output_image = overlay_mask(output_image, mask_2d, color=(255, 0, 0), alpha=0.7)

                        output_pil = Image.fromarray(output_image)
                        draw = ImageDraw.Draw(output_pil)  # draw 다시 초기화 필요 (PIL 객체 변경됐기 때문)

                        if idx in [31]:
                            xs = [p[0] for p in upanddown_as_tuples[idx]]
                            ys = [p[1] for p in upanddown_as_tuples[idx]]
                            rect_x_min, rect_x_max = min(xs), max(xs)
                            rect_y_min, rect_y_max = min(ys), max(ys)
                            draw.rectangle(
                                [(rect_x_min, rect_y_min), (rect_x_max, rect_y_max)],
                                outline="yellow", width=1
                            )

                        # 텍스트 출력
                        status = ""
                        if isPixelOverlappingLeft == False:
                            status += "L"
                        if isPixelOverlappingRight == False:
                            status += "R" 
                        if isPixelOverlappingUp == False:
                            status += "U"
                        if isSpecialCase == True:
                            status += "S"
                        

                        # 텍스트 출력
                        x, y = pt
                        draw.text((x-10, y - 70), f"{score_val*100:.0f}", fill="yellow", font=font30)
                        draw.text((x-10, y - 40), f"{idx}", fill="green", font=font)
                        draw.text((x-10, y - 25), f"{status}", fill="yellow", font=font)
                        draw.text((x-10, y - 10), f"{edge_cnt}", fill="yellow", font=font_edge)
                        if wh_test == False:
                            draw.text((x-30, y + 5), f"{width:.0f},{height:.0f}", fill="red", font=font)   

                        output_image = np.array(output_pil)    

                    else:
                        output_pil = Image.fromarray(output_image)
                        draw = ImageDraw.Draw(output_pil)  # draw 다시 초기화 필요 (PIL 객체 변경됐기 때문)

                        # 텍스트 출력
                        x, y = pt
                        status = ""
                        if isPixelOverlappingLeft == False:
                            status += "L"
                        if isPixelOverlappingRight == False:
                            status += "R" 
                        if isPixelOverlappingUp == False:
                            status += "U"
                        if isSpecialCase == True:
                            status += "S"
                        

                        # 텍스트 출력
                        x, y = pt
                        draw.text((x-10, y - 70), f"{score_val*100:.0f}", fill="red", font=font30)
                        draw.text((x-10, y - 40), f"{idx}", fill="green", font=font)
                        draw.text((x-10, y - 25), f"{status}", fill="yellow", font=font)
                        draw.text((x-10, y - 10), f"{edge_cnt}", fill="yellow", font=font_edge)
                        if wh_test == False:
                            draw.text((x-25, y + 10), f"{width:.0f},{height:.0f}", fill="red", font=font)   

                        output_image = np.array(output_pil)   

            output_pil = Image.fromarray(output_image)
            draw = ImageDraw.Draw(output_pil)   
            draw.text((513, 410), f"{occupiedCount:.0f} / {emptyCount:.0f}", fill="yellow", font=font100)    
            output_image = np.array(output_pil)   

            file_name = os.path.basename(filename)
            name, ext = os.path.splitext(file_name)  # name = "image001", ext = ".jpg"
            file_name_with_count = f"{name}_{occupiedCount}{ext}"

            save_path = os.path.join(folder_completed_path, file_name_with_count)
            Image.fromarray(output_image.astype(np.uint8)).save(save_path)

            print(f"{filename}, saved to {save_path}")
            write_log(f"{filename}, saved to {save_path}")
            write_log(f"\n\n")


check_parking_slot_using_image()
