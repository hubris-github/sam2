import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn.functional as F

def build_model(device, weights_path):
    # 1) 사전학습된 ResNet18 불러와서 fc 레이어 교체
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)   # binary logit
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model

def build_model2(device, weights_path):
    # 1) 사전학습된 ResNet18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features

    # ↓ checkpoint에 맞춰 2개의 출력을 가지도록 정의
    model.fc = nn.Linear(num_ftrs, 2)

    # 2) 저장된 state_dict 로드
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    return model.to(device).eval()

# 2) 학습 때 썼던 것과 동일한 transform
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # ResNet18 기본 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path, model, device, threshold=0.5):
    # 3) 이미지 열고 전처리
    img = Image.open(image_path).convert("RGB")
    inp = inference_transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    # 4) Forward → Sigmoid → Threshold
    with torch.no_grad():
        logit = model(inp)                # shape (1,1)
        prob  = torch.sigmoid(logit).item()
    pred_label = 'car' if prob > threshold else 'no_car'
    return pred_label, prob

def predict_image2(image_path, model, device):
    img = Image.open(image_path).convert("RGB")
    inp = inference_transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(inp)              # shape: [1, 2]
        probs  = F.softmax(logits, dim=1)  # shape: [1, 2]
        prob_car    = probs[0, 0].item()   # class 0 = car
        prob_no_car = probs[0, 1].item()   # class 1 = no_car
        pred_idx = probs.argmax(dim=1).item()

    label = 'car' if pred_idx == 0 else 'no_car'
    confidence = prob_car if pred_idx == 0 else prob_no_car
    return label, confidence

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 저장된 가중치 파일(.pth)의 경로
    # weights_path = 'best_resnet18_01.pth'
    # model = build_model(device, weights_path)

    # test_image = 'D:/Projects/vision/dataset/test/car_0001.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"차량 예측 결과 : {label} (confidence: {confidence:.3f})")

    # test_image = 'D:/Projects/vision/dataset/test/nocar_0001.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"주차면 예측 결과 : {label} (confidence: {confidence:.3f})")

    # weights_path = 'best_resnet18_02.pth'
    # model = build_model(device, weights_path)

    # test_image = 'D:/Projects/vision/dataset2/test/car_0000.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"차량 예측 결과 : {label} (confidence: {confidence:.3f})")

    # test_image = 'D:/Projects/vision/dataset2/test/nocar_0163.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"주차면 예측 결과 : {label} (confidence: {confidence:.3f})")

    # weights_path = 'best_resnet18_03.pth'
    # model = build_model(device, weights_path)

    # test_image = 'D:/Projects/vision/dataset3/test/car_0000.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"차량 예측 결과 : {label} (confidence: {confidence:.3f})")

    # test_image = 'D:/Projects/vision/dataset3/test/car_0096.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"차량 예측 결과 : {label} (confidence: {confidence:.3f})")


    # test_image = 'D:/Projects/vision/dataset3/test/nocar_0163.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"주차면 예측 결과 : {label} (confidence: {confidence:.3f})")

    # test_image = 'D:/Projects/vision/dataset3/test/nocar_0807.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"주차면 예측 결과 : {label} (confidence: {confidence:.3f})")

    # weights_path = 'best_resnet18_04_gray.pth'
    # model = build_model(device, weights_path)

    # test_image = 'D:/Projects/vision/dataset4/test/car_0000.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"차량 예측 결과 : {label} (confidence: {confidence:.3f})")

    # test_image = 'D:/Projects/vision/dataset4/test/nocar_0163.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"주차면 예측 결과 : {label} (confidence: {confidence:.3f})")


    # weights_path = 'best_resnet18_05.pth'
    # model = build_model(device, weights_path)

    # test_image = 'D:/Projects/vision/dataset5/test/car_0000.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"차량 예측 결과 : {label} (confidence: {confidence:.3f})")

    # test_image = 'D:/Projects/vision/dataset5/test/nocar_0163.jpg'
    # label, confidence = predict_image(test_image, model, device)

    # print(f"입력 이미지: {test_image}")
    # print(f"주차면 예측 결과 : {label} (confidence: {confidence:.3f})")

    weights_path = 'best_resnet18_07_black.pth'
    model = build_model2(device, weights_path)

    test_image = 'D:/Projects/vision/dataset7/test/car_00000.jpg'
    label, confidence = predict_image2(test_image, model, device)

    print(f"입력 이미지: {test_image}")
    print(f"차량 예측 결과 : {label} (confidence: {confidence:.3f})")

    test_image = 'D:/Projects/vision/dataset7/test/nocar_00000.jpg'
    label, confidence = predict_image2(test_image, model, device)

    print(f"입력 이미지: {test_image}")
    print(f"주차면 예측 결과 : {label} (confidence: {confidence:.3f})")