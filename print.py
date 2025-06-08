import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False, borders=True):
    """
    주어진 바이너리 마스크를 투명 컬러로 이미지 위에 표시합니다.
    
    매개변수:
    - mask (np.ndarray): 높이(H) x 너비(W) 크기의 바이너리(0 또는 1) 마스크 배열
    - ax (matplotlib.axes.Axes): 마스크를 그릴 matplotlib 축 객체
    - random_color (bool): True면 매번 랜덤 컬러를, False면 고정된 파란색 계열(투명 포함) 사용
    - borders (bool): True면 마스크 외곽선을 그려줌
    """
    # 1) 마스크 위에 얹을 색상 결정
    if random_color:
        # np.random.random(3)으로 0~1 사이의 랜덤 RGB 값을 생성하고,
        # 마지막에 투명도(alpha) 0.6을 넣어서 총 4차원(RGBA) 벡터 생성
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        # 예시: [0.23, 0.76, 0.10, 0.6]
    else:
        # 고정된 색상을 사용 (파란색 계열)
        # [R, G, B, A] 각각 0~1 사이 값. 30/255 ≈ 0.1176, 144/255 ≈ 0.5647, 255/255 = 1.0, 투명도 0.6
        color = np.array([30/255, 144/255, 255/255, 0.6])

    # 2) mask.shape[-2:], 즉 mask.shape이 (H, W)라고 가정
    h, w = mask.shape[-2:]  # h=높이, w=너비

    # 3) mask를 uint8 타입(0 또는 1)으로 변환
    #    (이미 바이너리라면 값이 0 또는 1이므로 astype으로 타입만 변경)
    mask = mask.astype(np.uint8)

    # 4) mask.reshape(h, w, 1): (H, W, 1) 형태로 바꾸고,
    #    여기에 color.reshape(1, 1, -1) 을 곱해서 (H, W, 4) RGBA 마스크 이미지 생성
    #    예를 들어 mask[i,j]=1인 픽셀은 color RGBA가 적용되고, 0인 픽셀은 (0,0,0,0) 값
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #    결과: mask_image.shape == (H, W, 4), 값 범위는 0~1.

    # 5) 윤곽선(border)을 그리고 싶다면 OpenCV를 사용
    if borders:
        import cv2  # OpenCV 라이브러리
        # cv2.findContours: 바이너리 이미지(mask)에서 외곽선을 검출
        # cv2.RETR_EXTERNAL: 외곽선 중 가장 바깥쪽만 (내부 중첩 윤곽선 무시)
        # cv2.CHAIN_APPROX_NONE: 모든 윤곽점 저장(압축 없음)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours는 리스트 형태로, 각각이 (N_i, 1, 2) 모양의 윤곽점 좌표 배열

        # 윤곽선을 좀 더 부드럽게 다듬기 위해 근사화(approxPolyDP) 수행
        # epsilon=0.01: 근사화 허용 오차. 이 값이 작을수록 원본 윤곽에 가까움
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
            for contour in contours
        ]
        # 그 결과 contours 안의 각 윤곽선이 좀 더 단순화된 좌표 배열이 됨

        # cv2.drawContours: mask_image 위에 윤곽선을 그림
        # arguments:
        #   mask_image: 그릴 대상 이미지 (float32로 되어 있어도 동작)
        #   contours: 위에서 근사화한 윤곽선 리스트
        #   -1: 모든 contour를 그림
        #   (1, 1, 1, 0.5): RGBA 값 (흰색 선, 투명도 0.5)
        #   thickness=2: 윤곽선 두께 2픽셀
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    # 6) 완성된 mask_image를 matplotlib 축(ax)에 보여줌
    #    이렇게 하면 원래 배경 이미지 위에 RGBA 마스크가 반투명으로 오버레이됨
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    사용자 클릭(포인트) 정보(양성/음성)를 이미지 위에 점으로 표시합니다.
    
    매개변수:
    - coords (np.ndarray): shape (N, 2)인 배열, 각 행은 (y, x) 형태의 좌표
    - labels (np.ndarray): shape (N,)인 배열, 1은 foreground(양성) 포인트, 0은 background(음성) 포인트
    - ax (matplotlib.axes.Axes): 점을 찍을 matplotlib 축 객체
    - marker_size (int): 점 크기 (잡아당겨보기를 편하게 크게 설정)
    """
    # coords[labels == 1] → 양성(positives) 포인트 좌표들만 선택
    pos_points = coords[labels == 1]
    # coords[labels == 0] → 음성(negatives) 포인트 좌표들만 선택
    neg_points = coords[labels == 0]

    # 1) 양성 포인트는 초록색 별표(*) 마커로 그리기
    #   s=marker_size: 마커 크기, edgecolor='white': 흰 테두리, linewidth=1.25: 테두리 두께
    if pos_points.size > 0:
        ax.scatter(
            pos_points[:, 0],  # y 좌표 (vertial 좌표)
            pos_points[:, 1],  # x 좌표 (horizontal 좌표)
            color='green',     # 마커 색상: 초록
            marker='*',        # 별표 모양
            s=marker_size,     # 마커 크기
            edgecolor='white', # 경계선 흰색
            linewidth=1.25     # 경계선 두께
        )
    # 2) 음성 포인트는 빨간색 별표(*) 마커로 그리기
    if neg_points.size > 0:
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color='red',       # 마커 색상: 빨강
            marker='*',
            s=marker_size,
            edgecolor='white',
            linewidth=1.25
        )


def show_box(box, ax):
    """
    사용자 지정 바운딩 박스(사각형)를 이미지 위에 녹색 외곽선으로 표시합니다.
    
    매개변수:
    - box (list or tuple or np.ndarray): [x_min, y_min, x_max, y_max] 형태
    - ax (matplotlib.axes.Axes): 사각형을 그릴 matplotlib 축 객체
    """
    # 좌상단(x0, y0) = (box[0], box[1])
    x0, y0 = box[0], box[1]
    # 너비(w) = x_max - x_min, 높이(h) = y_max - y_min
    w = box[2] - box[0]
    h = box[3] - box[1]

    # 사각형 패치(Rectangle)를 생성하여 축에 추가
    # edgecolor='green': 테두리 색상 초록, facecolor=(0,0,0,0): 내부 투명(아무 색 없음), lw=2: 선 두께
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    원본 이미지 위에 여러 마스크와 포인트, 바운딩 박스를 차례대로 보여주는 함수입니다.
    각 마스크마다 별도의 그림을 그립니다.
    
    매개변수:
    - image (np.ndarray 또는 PIL 이미지): 원본 이미지 (H_img x W_img x 3 또는 x4) 배열
    - masks (np.ndarray): shape (M, H_mask, W_mask)인 마스크 배열, M은 마스크 개수
    - scores (iterable): length M인 리스트나 배열로, 각 마스크의 신뢰도 점수
    - point_coords (np.ndarray, 선택적): shape (N, 2) (y, x) 형태 포인트 좌표 배열
    - box_coords (list, tuple, 선택적): [x_min, y_min, x_max, y_max] 형태 바운딩 박스 좌표
    - input_labels (np.ndarray, 선택적): shape (N,)인 포인트 레이블(1=양성, 0=음성)
    - borders (bool): show_mask 내부에서 윤곽선을 그릴지 여부
    """
    # masks, scores가 같은 길이(M)라고 가정
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # 1) 새 Figure와 축(ax) 생성. figsize=(10,10)은 큰 이미지로 표시하려는 설정
        plt.figure(figsize=(10, 10))
        # 2) 원본 이미지 보여주기
        plt.imshow(image)
        # 3) 디버깅용 출력: 현재 마스크의 shape 확인
        print(f"Mask shape: {mask.shape}")
        # 4) show_mask 함수 호출: 마스크를 투명 컬러로 오버레이
        show_mask(mask, plt.gca(), random_color=True, borders=borders)

        # 5) 포인트가 주어진 경우, 포인트 표시
        if point_coords is not None:
            # input_labels가 None이면 오류 발생. 포인트 레이블 없으면 점을 그릴 수 없음
            assert input_labels is not None, "포인트 레이블(input_labels)을 함께 제공해야 합니다."
            show_points(point_coords, input_labels, plt.gca(), marker_size=375)

        # 6) 바운딩 박스가 주어진 경우, 박스 표시
        if box_coords is not None:
            show_box(box_coords, plt.gca())

        # 7) 여러 개의 마스크가 있으면, 제목으로 "Mask i, Score ..." 출력
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)

        # 8) 축(axis) 표시 끄기 (눈금 제거)
        plt.axis('off')
        # 9) 화면에 이미지와 오버레이된 마스크, 포인트, 박스를 출력
        plt.show()
