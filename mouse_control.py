import HandTrackingModule as ht
from lib import *
import autopy  # Install using "pip install autopy"
import pyautogui
import voice_input


def ms_controller(detector, fingers, lmList, width, height, frameR, smoothening, screen_width, screen_height, img,
                  prev_x, prev_y, curr_x, curr_y):
    length_left, _, img = detector.findDistance(lmList[4][0:2], lmList[8][0:2], img)  # 좌클릭, 엄지 검지 거리
    length_right, _, img = detector.findDistance(lmList[4][0:2], lmList[12][0:2], img)  # 우클릭, 엄지 중지 거리
    length_center, info, img = detector.findDistance(lmList[0][0:2], lmList[9][0:2], img)  # 마우스 이동, 손바닥 중심

    x3 = np.interp(info[4], (frameR, width - frameR), (0, screen_width))
    y3 = np.interp(info[5], (frameR, height - frameR), (0, screen_height))

    cv2.circle(img, (info[4], info[5]), 15, (255, 0, 0), cv2.FILLED)
    curr_x = prev_x + (x3 - prev_x) / smoothening
    curr_y = prev_y + (y3 - prev_y) / smoothening
    d_y = prev_y - curr_y

    autopy.mouse.move(curr_x, curr_y)
    prev_x, prev_y = curr_x, curr_y

    if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
        pyautogui.scroll(int(d_y) + 100) if int(d_y) >= 0 else pyautogui.scroll(int(d_y) - 100)
    elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        voice_input.input_text()
    else:
        if length_left <= 20:
            pyautogui.leftClick()
        if length_right <= 30:
            pyautogui.rightClick()

    return prev_x, prev_y, curr_x, curr_y