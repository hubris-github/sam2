from PIL import ImageGrab
from PIL import ImageGrab
import time
import datetime

# 1) 캡처 전 2초 대기
time.sleep(2)

# 1. 화면 캡처
bbox = (115, 210, 1890, 1200)
screenshot = ImageGrab.grab(bbox=bbox)

folder = "D:/Projects/react/aiparking/public/live/"

# 2. 파일로 저장
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = folder + f"screenshot.png"
screenshot.save(filename)

print(f"Saved screenshot to {filename}")
