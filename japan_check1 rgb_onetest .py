
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


def overlap_pixels(rect_points, mask_uint8):
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
    # 1) 좌표를 int32 numpy 배열로 변환
    pts = np.array(rect_points, dtype=np.int32)

    # 2) mask와 동일 크기의 빈 마스크 생성
    poly_mask = np.zeros_like(mask_uint8, dtype=np.uint8)

    # 3) 다각형 내부를 1로 채우기
    cv2.fillPoly(poly_mask, [pts], 1)

    # 4) AND 연산하여 겹치는 부분만 남기고, 픽셀 수 세기
    intersection = mask_uint8 & poly_mask
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

file_template = "1_*.png" # 1_20250711_235044sshot
#folder_path = 'D:/Projects/vision/yolo/images/mp4/japan/'
folder_path = 'D:/Projects/vision/capture_images/20250710/'
folder_completed_path = 'D:/Projects/vision/capture_images/20250710/completed8/'

start_index = 0
end_index = 0# 1443
index_step = 1

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
  
    pattern = folder_path + "1_*sshot.png"
    files = glob.glob(pattern)

    for filename in files[start_index : end_index + 1 : index_step]:  

        # folder_file = folder_path + filename
        filename = folder_path + "1_20250711_012904sshot.png"
        filename = folder_path + "1_20250711_011059sshot.png"
        
        print(f"Processing file: {filename}")

        if os.path.exists(filename):
            image_pil = Image.open(filename)
            image_np = np.array(image_pil.convert("RGB"))

            output_image = image_np.copy()  # 마스크 누적할 이미지

            output_pil = Image.fromarray(output_image)
            draw = ImageDraw.Draw(output_pil)

            try:
                    font = ImageFont.truetype("arial.ttf", 15)  # 시스템 폰트가 있으면 사용
                    font2 = ImageFont.truetype("arial.ttf", 100)
            except:
                    font = ImageFont.load_default()
                    font2 = ImageFont.load_default()

            # print("cwd:", os.getcwd(), "latest file:", filename)
            # print("image shape:", image.shape)

            predictor.set_image(image_np)

            input_points = np.array([
                [14, 556],  #1 (idx 0)
                [45, 612],  #2
                [63, 626],  #3
                [123, 633], #4
                [190, 688], #5
                [262, 697], #6
                [384, 724], #7
                [500, 727], #8
                [646, 744], #9
                [793, 757], #10
                [922, 736], #11
                [1054, 696], #12
                [1149, 696], #13
                [1237, 659], #14
                [1324, 670], #15 
                [1382, 652], #16 (idx 15)
                [1405, 632], #17
                [1400, 567], #18

                [160, 435],  #19 (1)
                [188, 442],  #20 (2)
                [245, 455],  #21 (3)
                [308, 453],  #22 (4)  
                [371, 445],  #23 (5)
                [445, 451],  #24 (6)
                [513, 440],  #25 (7)
                [603, 456],  #26 (8)
                [687, 461],  #27 (9)
                [776, 462],  #28 (10)
                [862, 465],  #29 (11)
                [948, 453],  #30 (12)
                [1020, 463], #31 (13)
                [1085, 461], #32 (14)
                [1158, 464], #33 (15)
                [1211, 465], #34 (16)
                [1266, 454], #35 (17)
                [1307, 459], #36 (18)
                [1340, 445], #37 (19) (idx 36)
            ])
            
            input_label = np.array([1])  # 양성(1)으로 설정

            # print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

            occupiedCount = 0 
            emptyCount = 0

            #                   0      1      2      3      4      5      6      7      8       9     10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36     37
            #                   1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36     37     38
            #                   A1     A2     A3     A4     A5     A6     A7     A8     A9     B1     B2     B3     B4     B5     B6     B7     B8     B9     B10    B11    B12    C1     C2     C3     C4     C5     C6     C7     C8     C9     C10    C11    C12    D1     D2     D3     D4     D5
            lower_thresholds = [2000,  5000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3000,  3500,  3500,  3500,  3500,  3000,  2500,  1500,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  1500 ]
            upper_thresholds = [10000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 30000, 30000, 30000, 30000, 30000]
            score_thresholds = [0.7,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.4,   0.6,   0.5,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6,   0.6  ]
            wh_thresholds    = [1,     1,     1,     2,     2,     2,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     2,     2,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1    ]
            vehicleDetected  = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
            width_thresholds = [90,    160,   160,   160,   160,   160,   160,   140,   150,   130,   150,   150,   170,   145,   140,   130,   120,   70,    110,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   110,   110,   0    ]
            height_thresholds= [160,   160,   160,   160,   160,   160,   160,   160,   220,   220,   220,   220,   170,   150,   150,   150,   150,   100,   110,   130,   130,   140,   140,   140,   140,   150,   150,   150,   150,   140,   140,   140,   130,   130,   120,   120,   120,   0    ]
            wh_direction     = [-1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    0,     0,     0,     1,     1,     1,     1,     1,     1,     1,     -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    0,     0,     0,     1,     1,     1,     1,     1,     1,     1,     1,     1    ]
            slots = [f"A{i}" for i in range(1, 13)] + [f"B{i}" for i in range(1, 13)]
            rectangles_as_tuples = [
                [(53, 529), (1, 581), (11, 614), (78, 538)],      # 1 (idx 0)
                [(88, 542), (18, 619), (42, 634), (121, 549)],    # 2
                [(127, 554), (53, 641), (85, 658), (166, 562)],   # 3
                [(182, 564), (98, 665), (124, 675), (207, 576)],  # 4
                [(236, 578), (153, 691), (209, 712), (293, 590)], # 5 (idx 4)
                [(311, 596), (227, 719), (304, 742), (383, 607)], # 6
                [(398, 606), (323, 746), (416, 763), (478, 616)], # 7
                [(502, 619), (450, 747), (558, 748), (589, 625)], # 8
                [(617, 629), (580, 796), (698, 807), (712, 633)], # 9
                [(737, 633), (735, 801), (853, 802), (833, 634)], #10
                [(860, 632), (889, 802), (996, 791), (950, 630)],     # 11
                [(974, 628), (1026, 786), (1117, 767), (1053, 623)],  # 12
                [(1075, 620), (1141, 762), (1215, 743), (1144, 613)], # 13
                [(1162, 609), (1234, 736), (1292, 717), (1218, 602)], # 14
                [(1234, 599), (1308, 711), (1350, 693), (1280, 591)], # 15
                [(1293, 588), (1362, 686), (1395, 669), (1330, 579)], # 16
                [(1339, 575), (1405, 664), (1428, 649), (1371, 570)], # 17
                [(1379, 566), (1428, 630), (1429, 594), (1401, 560)], # 18(우끝)
                
                [(198, 402), (138, 455), (167, 457), (228, 405)], # 19
                [(233, 404), (175, 458), (210, 459), (272, 408)], # 20
                [(281, 409), (220, 462), (260, 466), (318, 411)], # 21
                [(331, 411), (271, 467), (319, 468), (372, 410)], # 22
                [(386, 411), (333, 470), (385, 472), (434, 414)], # 23
                [(450, 416), (399, 475), (459, 477), (502, 415)], # 24
                [(518, 414), (477, 479), (540, 481), (572, 416)], # 25
                [(590, 418), (559, 484), (629, 483), (647, 418)], # 26
                [(665, 419), (649, 485), (720, 487), (724, 421)], # 27
                [(743, 421), (741, 488), (811, 489), (802, 422)], # 28
                [(820, 422), (833, 488), (901, 489), (877, 423)], # 29
                [(895, 423), (921, 489), (985, 488), (956, 421)], # 30
                [(972, 424), (1003, 488), (1063, 488), (1020, 426)], # 31
                [(1033, 425), (1078, 486), (1131, 487), (1080, 426)], # 32
                [(1095, 426), (1146, 485), (1191, 485), (1136, 427)], # 33
                [(1150, 426), (1203, 482), (1241, 481), (1186, 427)], # 34
                [(1199, 429), (1253, 480), (1287, 480), (1229, 428)], # 35
                [(1252, 429), (1303, 479), (1348, 476), (1296, 427)], # 36
                [(1297, 423), (1353, 473), (1374, 471), (1326, 426)], # 37
                [(1326, 425), (1374, 471), (1400, 467), (1356, 422)], # 38 (임시로)
            ]

            upanddown_as_tuples = [
                [(0, 0), (0, 0), (0, 0), (0, 0)], #1 (0)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #2 (1)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #3 (2)                
                [(0, 0), (0, 0), (0, 0), (0, 0)], #4 (3)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #5 (4)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #6 (5)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #7 (6)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #8 (7)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #9 (8)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #10 (9)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #11 (10)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #12 (11)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #13 (12)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #14 (13)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #15 (14)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #16 (15)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #17 (16)
                [(0, 0), (0, 0), (0, 0), (0, 0)], #18 (17)
                
                [(166, 371), (166, 402), (182, 402), (182, 371)], # 19 (18)
                [(203, 371), (203, 404), (225, 404), (225, 371)], # 20 (19)
                [(272, 371), (272, 403), (318, 403), (318, 371)], # 21 (20)
                [(315, 371), (315, 403), (337, 403), (337, 371)], # 22 (21)
                [(350, 371), (350, 403), (372, 403), (372, 371)], # 23 (22)
                [(424, 371), (424, 401), (455, 401), (455, 371)], # 24 (23)
                [(471, 371), (471, 404), (501, 404), (501, 371)], # 25 (24)
                [(579, 371), (579, 418), (623, 418), (623, 371)], # 26 (25)
                [(665, 371), (665, 419), (695, 419), (695, 371)], # 27 (26) 
                [(740, 371), (740, 406), (802, 406), (802, 371)], # 28 (27)
                [(825, 371), (825, 406), (850, 406), (850, 371)], # 29 (28)
                [(895, 371), (895, 406), (956, 406), (956, 371)], # 30 (29)
                [(976, 392), (976, 412), (1023,412), (1023,392)], # 31 (30) 
                [(1052,392), (1052,412), (1088,412), (1088,392)], # 32 (31) ##
                [(1104,395), (1104,411), (1155,411), (1155,395)], # 33 (32) 
                [(1187,398), (1187,411), (1195,411), (1195,398)], # 34 (33)
                [(1249,393), (1249,402), (1275,413), (1275,393)], # 35 (34)
                [(1300,397), (1300,417), (1304,417), (1304,397)], # 36 (35)
                [(1318,397), (1318,417), (1351,417), (1351,397)], # 37 (36)
                [(1354,397), (1354,417), (1374,417), (1374,397)], # 38 (37) (임시로)
            ]

            # idx 18 ~ 37(여기 18면에는 왼쪽 면이 없으므로 임시로 왼쪽면을 만들어 줌.)
            # idx 0 ~ 17 
            rectangles_as_tuples2 = [
                [(1352, 569), (1419, 657), (1429, 651), (1368, 568)], # idx 15(#16)
                [(1404, 556), (1427, 573), (1427, 558), (1422, 548)], # idx 17(#18)
                [(166, 388), (106, 447), (129, 451), (189, 405)], # idx 18(#19)
            ]


            second_point_for = {
                #12: [538, 276],  # idx=15(16번째) --> 두 번째 좌표는 [144, 303]
                #17: [1020, 303],  # idx=16(17번째) --> 두 번째 좌표는 [ 95, 277]
                 6: [420, 635],
                 9: [795, 660],
                16: [1385, 569],
                19: [250, 410],
                22: [400, 415],
                27: [770, 425],
                29: [936, 426],
                31: [1069, 431],
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
                selected_idx = 16
                if idx == selected_idx:
                    # np.savetxt("mask_2d.csv", mask_2d, fmt="%d", delimiter=",")
                    img = Image.fromarray((mask_2d * 255).astype(np.uint8))
                    img.save("mask_2d.png")

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
                    large_labels = np.where(counts > 500)[0]
                    large_labels = large_labels[large_labels != 0]  # 0 라벨(배경) 제거

                    # 4) 최종 필터링: 크기 30 이하 컴포넌트 제거
                    #    np.isin 으로 남길 라벨만 True 로
                    filtered_mask_clean = np.isin(labeled, large_labels)

                    # 필요 시 원래 형태(bool) 유지
                    filtered_mask_clean = filtered_mask_clean.astype(bool)

                    # 방법 1: np.where 로 새 배열 만들기
                    mask_2d = np.where(filtered_mask_clean, mask_2d, 0)

                    # print("#mask_2d shape:", mask_2d.shape, " dtype:", mask_2d.dtype)
                    # print("filtered mask_2d shape:", mask_2d.shape, "nonzero count:", np.count_nonzero(mask_2d))

                    # 방법 2: in-place 인덱싱 (원본 배열 변경)
                    # mask_2d = mask_2d.copy()            # 혹시 원본을 보존하고 싶다면
                    # mask_2d[~filtered_mask_clean] = 0

                    img = Image.fromarray((filtered_mask_clean * 255).astype(np.uint8))
                    img.save("mask_2d_filtered.png")

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
                isSpecialCase = False

                if idx > 0:     # 0번째는 왼쪽에 붙어 있으므로 제외, 1번째 인덱스부터 시작
                    mask_uint8 = mask_2d.astype(np.uint8)

                    if idx in range(1, 10) or idx in range(18, 26):  # 1~9, 18~25 영역의 자동차가 오른쪽 면을 침범하면 자동차가 아님(침범하면 보통 주차면임)
                        # 1) rect_right_idx: #2 영역의 좌표 (list of (x, y))
                        #overlap_pixels = overlap_pixels(rectangles_as_tuples[idx+1], mask_2d)
                        rect_right_idx = np.array(rectangles_as_tuples[idx+1], dtype=np.int32)
                        
                        # 2) 같은 크기의 빈 마스크 생성
                        poly_mask = np.zeros_like(mask_2d, dtype=np.uint8)

                        # 3) 사각형(다각형) 내부를 1로 채우기
                        cv2.fillPoly(poly_mask, [rect_right_idx], 1)

                        # 4) AND 연산하여 겹치는 픽셀 수 확인
                        intersection = mask_uint8 & poly_mask
                        overlap_pixels = np.count_nonzero(intersection)     # 겹치는 픽셀 수

                        print(f"idx+1: {idx+1}, (R1) overlap_pixels = {overlap_pixels}")

                        if overlap_pixels > 0:  # 오른쪽 주차면을 침범하면 안 될 경우, 침범하면 오류
                            # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                            isPixelOverlappingRight = False

                        # 주차면에 자동차가 있다면, 왼쪽으로 침범해야 함
                        if idx in range(1, 7) or idx in range(18, 24): #  # 1 ~ 6 영역의 자동차가 있으면 왼쪽으로 침범해야 함
                            # idx 18은 윗쪽의 맨 왼쪽 주차면이므로, rectangles_as_tuples2[2]을 사용
                            if idx == 18:
                                rect_left_idx = np.array(rectangles_as_tuples2[2], dtype=np.int32)
                            else:
                                rect_left_idx = np.array(rectangles_as_tuples[idx-1], dtype=np.int32)

                            # 2) 같은 크기의 빈 마스크 생성
                            poly_mask = np.zeros_like(mask_2d, dtype=np.uint8)

                            # 3) 사각형(다각형) 내부를 1로 채우기
                            cv2.fillPoly(poly_mask, [rect_left_idx], 1)

                            # 4) AND 연산하여 겹치는 픽셀 수 확인
                            intersection = mask_uint8 & poly_mask
                            overlap_pixels = np.count_nonzero(intersection)     # 겹치는 픽셀 수

                            print(f"idx+1: {idx+1}, (L1) overlap_pixels = {overlap_pixels}")

                            if overlap_pixels < 50:  # 왼쪽 주차면을 침범해야 함, 침범하지 않으면 주차된 차량이 없는 경우임
                                # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                                isPixelOverlappingLeft = False


                    elif idx in range(10, 18) or idx in range(28, 37):   # 10~17, 28~36 영역의 자동차가 왼쪽 면에 붙으면 안 됨
                        rect_left_idx = np.array(rectangles_as_tuples[idx-1], dtype=np.int32)

                        # 1) 같은 크기의 빈 마스크 생성
                        poly_mask = np.zeros_like(mask_2d, dtype=np.uint8)

                        # 2) 사각형(다각형) 내부를 1로 채우기
                        cv2.fillPoly(poly_mask, [rect_left_idx], 1)

                        # 3) AND 연산하여 겹치는 픽셀 수 확인
                        intersection = mask_uint8 & poly_mask
                        overlap_pixels = np.count_nonzero(intersection)

                        print(f"idx+1: {idx+1}, (L2) overlap_pixels = {overlap_pixels}")

                        if overlap_pixels > 0:  # 왼쪽 주차면을 침범하면 안 될 경우, 침범하면 오류
                            # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                            isPixelOverlappingLeft = False

                            # 왼쪽에 있는 차량이 색상이 같은 경우 왼쪽으로 침범하는 경우가 있음.
                            if idx in {13, 15} and vehicleDetected[idx-1]:
                                # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                                isSpecialCase = True    


                        # 주차면에 자동차가 있다면, 오른쪽으로 침범해야 함
                        if idx in range(10, 18) or idx in range(31, 37): #  # 10 ~ 17, 31 ~ 37 영역의 자동차가 있으면 오른쪽으로 침범해야 함
                            # 1) rect_right_idx: #2 영역의 좌표 (list of (x, y))
                            
                            if idx == 17:  # idx 17은 윗쪽의 맨 오른쪽 주차면이므로, rectangles_as_tuples2[1]을 사용
                                rect_right_idx = np.array(rectangles_as_tuples2[1], dtype=np.int32) 
                            else:
                                rect_right_idx = np.array(rectangles_as_tuples[idx+1], dtype=np.int32)

                            # 2) 같은 크기의 빈 마스크 생성
                            poly_mask = np.zeros_like(mask_2d, dtype=np.uint8)

                            # 3) 사각형(다각형) 내부를 1로 채우기
                            cv2.fillPoly(poly_mask, [rect_right_idx], 1)

                            # 4) AND 연산하여 겹치는 픽셀 수 확인
                            intersection = mask_uint8 & poly_mask
                            overlap_pixels = np.count_nonzero(intersection)     # 겹치는 픽셀 수

                            print(f"idx+1: {idx+1}, (R2) overlap_pixels = {overlap_pixels}")

                            if overlap_pixels == 0:  # 오른쪽 주차면을 침범해야 함, 침범하지 않으면 주차된 차량이 없는 경우임
                                # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                                isPixelOverlappingRight = False
                            elif idx == 12:
                                
                                if overlap_pixels < 300:  # 오른쪽 주차면을 300픽셀 이상 침범해야 함, 침범하지 않으면 주차된 차량이 없는 경우임
                                    # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                                    isPixelOverlappingRight = False
                                    print(f"idx+1: {idx+1}, (IDX12 (R) overlap_pixels = {overlap_pixels}")

                            elif idx == 15:
                                
                                rect_right_idx = np.array(rectangles_as_tuples2[0], dtype=np.int32) 

                                # 2) 같은 크기의 빈 마스크 생성
                                poly_mask = np.zeros_like(mask_2d, dtype=np.uint8)

                                # 3) 사각형(다각형) 내부를 1로 채우기
                                cv2.fillPoly(poly_mask, [rect_right_idx], 1)

                                # 4) AND 연산하여 겹치는 픽셀 수 확인
                                intersection = mask_uint8 & poly_mask
                                overlap_pixels = np.count_nonzero(intersection)     # 겹치는 픽셀 수

                                print(f"idx+1: {idx+1}, (IDX15) (R) overlap_pixels = {overlap_pixels}")

                                if overlap_pixels == 0:  # 왼쪽 주차면을 침범해야 함, 침범하지 않으면 주차된 차량이 없는 경우임
                                    # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                                    isPixelOverlappingRight = False
                                
                                
                            # 한 주차면에서 차량 검출 시 3개의 주차면을 차지하면 오류
                            if idx == 16:  # idx 16에서 17의 오른쪽 영역(18)을 침범하면 안 됨, rectangles_as_tuples2[2]을 사용

                                rect_right_idx = np.array(rectangles_as_tuples2[1], dtype=np.int32) 

                                # 2) 같은 크기의 빈 마스크 생성
                                poly_mask = np.zeros_like(mask_2d, dtype=np.uint8)

                                # 3) 사각형(다각형) 내부를 1로 채우기
                                cv2.fillPoly(poly_mask, [rect_right_idx], 1)

                                # 4) AND 연산하여 겹치는 픽셀 수 확인
                                intersection = mask_uint8 & poly_mask
                                overlap_pixels = np.count_nonzero(intersection)     # 겹치는 픽셀 수

                                print(f"idx+1: {idx+1}, (IDX16) (R) overlap_pixels = {overlap_pixels}")

                                if overlap_pixels > 500:  # 2칸 건너 주차면을 침범하면 안 됨
                                    # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
                                    isPixelOverlappingRight = False




                    # 자동차가 검출되었으면 윗쪽 영역을 침범해야 함                
                    if idx in range(18, 37):        # 18~36 자동차가 있으면 주차선 윗쪽으로 자동차가 침범해야 함
                        rect_up_idx = np.array(upanddown_as_tuples[idx], dtype=np.int32)

                        # 2) 같은 크기의 빈 마스크 생성
                        poly_mask = np.zeros_like(mask_2d, dtype=np.uint8)

                        # 3) 사각형(다각형) 내부를 1로 채우기
                        cv2.fillPoly(poly_mask, [rect_up_idx], 1)

                        # 4) AND 연산하여 겹치는 픽셀 수 확인
                        intersection = mask_uint8 & poly_mask
                        overlap_pixels = np.count_nonzero(intersection)     # 겹치는 픽셀 수

                        print(f"idx+1: {idx+1}, (U) overlap_pixels = {overlap_pixels}")

                        if overlap_pixels == 0:  # 왼쪽 주차면을 침범해야 함, 침범하지 않으면 주차된 차량이 없는 경우임
                            # print(f"idx: {idx+1}, ⚠️ 겹침 발생: {overlap_pixels} 픽셀")
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

                wh_test = width < width_thresholds[idx] and height < height_thresholds[idx] 

                # (5) mask_2d를 bool로 변환한 뒤, filtered_mask_clean과 AND 연산
                mask_bool = mask_2d.astype(bool) & filtered_mask_clean

                # bool 마스크를 이용해 원본 이미지에서 픽셀 추출
                masked_pixels = image_np[mask_bool]  # shape: (N, 3)

                # masked_pixels = image_np[mask_2d.astype(bool)]  # shape: (N, 3)
                unique_colors = np.unique(masked_pixels, axis=0)
                color_count = len(unique_colors)

                # (선택) masked_pixels가 비어 있으면 cvtColor 호출을 건너뛰도록 방어
                if masked_pixels.size == 0:
                    hsv_pixels = np.zeros((0, 3), dtype=np.uint8)
                else:
                    hsv_pixels = (
                        cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
                        .reshape(-1, 3)
                    )

                # hsv_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                saturation_std = hsv_pixels[:, 1].std()

                aspect_ratio = width / height if height != 0 else 0
                y_center = ys.mean() if len(ys) > 0 else 0

                print(f"=" * 120)
                print(f"idx+1: {idx+1}, pixel_count               = {pixel_count}")
                print(f"idx+1: {idx+1}, score_val >= score_thresh = {score_val >= score_thresh}, {score_val}, {score_thresh}")
                print(f"idx+1: {idx+1}, pixel_count > lower       = {pixel_count > lower}, {pixel_count}, {lower}")
                print(f"idx+1: {idx+1}, pixel_count < upper       = {pixel_count < upper}, {pixel_count}, {upper}")
                print(f"idx+1: {idx+1}, isPixelOverlapping(LRUS)  = {isPixelOverlappingLeft}, {isPixelOverlappingRight}, {isPixelOverlappingUp}, {isSpecialCase}")
                print(f"idx+1: {idx+1}, wh_test                   = {wh_test}, width={width}, height={height}")
                print(f"idx+1: {idx+1}, x, y, xmax, ymax          = {x_min}, {y_min}, {x_max}, {y_max}")
                print(f"idx+1: {idx+1}, 색상수, 채도편차          = {color_count}, {saturation_std:.2f}")
                print(f"idx+1: {idx+1}, pt                        = {pt}")

                if score_val >= score_thresh and pixel_count > lower and pixel_count < upper and isPixelOverlappingUp and isPixelOverlappingRight and ((isPixelOverlappingLeft and wh_test) or isSpecialCase):
                    
                    vehicleDetected[idx] = True
                    print(f"-" * 120)
                    print(f"idx+1: {idx+1}, [# OOO #] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, Score={score_val:.3f}, width={width}, height={height}")
                    print(f"=-" * 60)
                    print(f" " * 10)
                    print(f" " * 10)
                    ### print(f"idx+1: {idx}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, 색상수={color_count}, 채도편차={saturation_std:.2f}, 비율={aspect_ratio:.2f}")

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

                    ## print(f"idx: {idx}, [O] Occupied Count: {occupiedCount}, 색상수={color_count}, 채도편차={saturation_std:.2f}, 비율={aspect_ratio:.2f}")# print(f"idx: {idx+1}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}, pixel_count={pixel_count}, saturation_std={saturation_std:.2f}, color_count={color_count}")
                    ## print(f"idx: {idx+1}, [O] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")
                else:
                    ## print(f"=" * 120)
                    ## print(f"idx+1: {idx+1}, score_val >= score_thresh={score_val >= score_thresh}")
                    ## print(f"idx+1: {idx+1}, pixel_count > lower      ={pixel_count > lower}, {pixel_count}, {lower}")
                    ## print(f"idx+1: {idx+1}, pixel_count < upper      ={pixel_count < upper}, {pixel_count}, {upper}")
                    ## print(f"idx+1: {idx+1}, wh_test                  ={wh_test}, width={width}, height={height}")                    
                    ## print(f"idx+1: {idx+1}, is_overlap_pixels        ={is_overlap_pixels}")
                    ## print(f"idx+1: {idx+1}, pt                       ={pt}")
                    print(f"-" * 120)
                    print(f"idx+1: {idx+1}, [# XXX #] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, Score={score_val:.3f}, width={width}, height={height}, 색상수={color_count}, 채도편차={saturation_std:.2f}, 비율={aspect_ratio:.2f}")
                    print(f"=-" * 60)
                    print(f" " * 10)
                    print(f" " * 10)


                    emptyCount += 1

                    if pixel_count < 50000:
                        if idx == selected_idx:
                            output_image = overlay_mask(output_image, mask_2d, color=(255, 0, 0), alpha=0.5)
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
                        
                        #for idx, rect in enumerate(rectangles_as_tuples):
                            # rect는 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 형태
                            # draw.polygon에 넘기면 네 점을 순서대로 연결하고 마지막 점에서 첫 점으로 닫혀 그림
                        
                        # draw.polygon(rectangles_as_tuples[idx], outline="green", width=10)

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

                        # draw.polygon(rectangles_as_tuples[idx], outline="green", width=10)

                        output_image = np.array(output_pil)   
                    # print(f"idx: {idx}, [X] Occupied Count: {occupiedCount}, 색상수={color_count}, 채도편차={saturation_std:.2f}, 비율={aspect_ratio:.2f}")
                    # print(f"idx: {idx+1}, [X] Occupied Count: {occupiedCount}, Empty Count: {emptyCount}, slot={slots[idx]}, Score={score_val:.3f}, width={width}, height={height}")

            output_pil = Image.fromarray(output_image)
            draw = ImageDraw.Draw(output_pil)   
            draw.text((600,  270), f"{occupiedCount:.0f} / {emptyCount:.0f}", fill="yellow", font=font2)    
            output_image = np.array(output_pil)   

            file_name = os.path.basename(filename)
            name, ext = os.path.splitext(file_name)  # name = "image001", ext = ".jpg"
            file_name_with_count = f"{name}_{occupiedCount}{ext}"

            save_path = os.path.join(folder_completed_path, file_name_with_count)
            #   cv2.imwrite(save_path, overlayed_image)
            
            #   os.makedirs("output", exist_ok=True)
            Image.fromarray(output_image.astype(np.uint8)).save(save_path)

            print(f"{filename}, saved to {save_path}")



check_parking_slot_using_image()
