import os
import re

# 대상 폴더 경로
target_dir = 'D:/Projects/vision/yolo/images/mp4/parquery_night/'
output_file = "D:/Projects/vision/yolo/images/mp4/file2.txt"

# 정규식 패턴: frame_숫자_33.jpg
pattern = re.compile(r"^frame_(\d{5,6})_(3[1-5])\.jpg$")
pattern = re.compile(r"^frame_(\d{5,6})_((?:[0-9]|[1-9][0-9]|3[0-5]))\.jpg$")

# 유효한 파일을 (숫자, 파일명) 형태로 수집
matched_files = []
for root, dirs, files in os.walk(target_dir):
    for filename in files:
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            matched_files.append((number, filename))

# 숫자 기준으로 정렬
matched_files.sort()

# 정렬된 파일명만 추출
sorted_filenames = [filename for _, filename in matched_files]

# 파일에 쓰기
with open(output_file, "w", encoding="utf-8") as f:
    for name in sorted_filenames:
        f.write(name + "\n")

print(f"{len(sorted_filenames)}개의 정렬된 파일명을 '{output_file}'에 저장했습니다.")

