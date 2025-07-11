import cv2
from gaze_tracking import GazeTracking
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

# ✅ 한글 텍스트 출력 함수 정의
def draw_text_korean(img, text, position, font_path, font_size=30, color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV → PIL
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # PIL → OpenCV

# ✅ 나눔고딕 폰트 경로 (사용자 환경에 맞게 수정)
font_path = "/Users/youth/Desktop/기본코드+패딩/GazeTracking/NanumGothicBold.otf"
if not os.path.isfile(font_path):
    raise FileNotFoundError(f"❌ 폰트 파일을 찾을 수 없습니다: {font_path}")

# ✅ GazeTracking 초기화
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # 시선 분석
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()
    text = ""

    # 시선 방향 판별
    if gaze.is_blinking():
        text = "깜빡임 감지"
    elif gaze.is_right():
        text = "오른쪽 응시"
    elif gaze.is_left():
        text = "왼쪽 응시"
    elif gaze.is_center():
        text = "정면 응시"

    # ✅ 한글 텍스트 출력
    annotated_frame = draw_text_korean(annotated_frame, text, (90, 60), font_path)

    # 동공 좌표 가져오기
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    # 동공 좌표 출력
    annotated_frame = draw_text_korean(annotated_frame, f"왼쪽 동공: {str(left_pupil)}", (90, 130), font_path, 24)
    annotated_frame = draw_text_korean(annotated_frame, f"오른쪽 동공: {str(right_pupil)}", (90, 165), font_path, 24)

    # 결과 출력
    cv2.imshow("GazeTracking + 한글 출력", annotated_frame)

    # ESC 키로 종료
    if cv2.waitKey(1) == 27:
        break

# 자원 해제
webcam.release()
cv2.destroyAllWindows()
