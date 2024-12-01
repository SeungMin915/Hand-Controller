from lib import *
import HandTrackingModule as ht
import FaceMeshModule as FM
import volumecontroller as vc
import mouse_control as mc
import autopy  # Install using "pip install autopy"
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from tensorflow.keras.models import load_model

### Variables Declaration
pTime = 0  # Used to calculate frame rate
width = 640  # Width of Camera
height = 480  # Height of Camera
frameR = 100  # Frame Rate
smoothening = 8  # Smoothening Factor
prev_x, prev_y = 0, 0  # Previous coordinates
curr_x, curr_y = 0, 0  # Current coordinates
before_direction = ""

# 카메라 지정 (0 : 노트북, 2: 웹캠)
cap = cv2.VideoCapture(0)  # Getting video feed from the webcam
cap.set(3, width)  # Adjusting size
cap.set(4, height)

# increase the confidence to increase precision
detector = ht.HandDetector(maxHands=2)  # Detecting one hand at max
facemesh = FM.faceDetector(max_num_faces=1)

screen_width, screen_height = autopy.screen.size()  # Getting the screen size

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

actions = ['give me', 'not give me']
seq_length = 30
model = load_model('models/model.h5')

seq = []
action_seq = []

handright1 = False  # 새로운 손
handright2 = False  # 기존의 손

before_CP = (0, 0)
is_len_2 = False

while True:
    if len(action_seq) > 100:
        action_seq = []

    success, img = cap.read()
    img = cv2.flip(img, 1)  # 웹캠 좌우반전
    hands, img = detector.findHands(img, draw=True, flipType=True)
    face_direction, img = facemesh.face_mseh_direction(before_direction, img)
    before_direction = face_direction

    if hands:
        # Hand 1
        if len(hands) == 1:
            if handright2 == True:
                handright2 == False
                action_seq = []

            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            if handright1 == True and is_len_2 == True:
                center_dis, _ = detector.findDistance(centerPoint1, before_CP)
                if center_dis > 50:
                    handright1 = False
                is_len_2 = False
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand1)

            if handright1 == False:
                handGesture1, img, actions, seq_length, model, seq, action_seq \
                    = detector.findGesture(img, actions=actions, seq_length=seq_length,
                                           model=model, seq=seq, action_seq=action_seq,  handlen=0)

                if handGesture1 == "give me":
                    handright1 = True
                    handright2 = False
                    action_seq = []
                    time.sleep(1)

            if handType1 == "Right" and handright1 == True \
                    and (face_direction == "Forward" or before_direction == "Forward"):
                prev_x, prev_y, curr_x, curr_y = mc.ms_controller(detector, fingers1, lmList1, width, height, frameR,
                                                                  smoothening, screen_width, screen_height, img, prev_x,
                                                                  prev_y, curr_x, curr_y)
            elif handType1 == "Left" and handright1 == True \
                    and (face_direction == "Forward" or before_direction == "Forward"):
                vc.vol_controller(lmList1, volume, volRange, img)

        elif len(hands) == 2:
            # Hand 2
            hand1 = hands[1]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right
            fingers1 = detector.fingersUp(hand1)
            if handright1 == False:
                handGesture1, img, actions, seq_length, model, seq, action_seq \
                    = detector.findGesture(img, actions=actions, seq_length=seq_length,
                                           model=model, seq=seq, action_seq=action_seq, handlen=1)

                if handGesture1 == "give me":
                    handright1 = True
                    handright2 = False
                    action_seq = []
                    time.sleep(1)
            if handright1 == True:
                is_len_2 = True
            hand2 = hands[0]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or move"Right"

            fingers2 = detector.fingersUp(hand2)
            if handright2 == False:
                handGesture2, img, actions, seq_length, model, seq, action_seq \
                    = detector.findGesture(img, actions=actions, seq_length=seq_length,
                                           model=model, seq=seq, action_seq=action_seq, handlen=0)

                if handGesture2 == "give me":
                    handright2 = True
                    handright1 = False
                    action_seq = []
                    time.sleep(1)

            if handType1 == "Left" and handright1 == True \
                    and (face_direction == "Forward" or before_direction == "Forward"):
                ##볼륨
                vc.vol_controller(lmList1, volume, volRange, img)
                before_CP = centerPoint1

            elif handType2 == "Left" and handright2 == True \
                    and (face_direction == "Forward" or before_direction == "Forward"):
                ## 볼륨
                vc.vol_controller(lmList2, volume, volRange, img)

            if handType1 == "Right" and handright1 == True \
                    and (face_direction == "Forward" or before_direction == "Forward"):
                prev_x, prev_y, curr_x, curr_y = mc.ms_controller(detector, fingers1, lmList1, width, height, frameR,
                                                                  smoothening, screen_width, screen_height, img, prev_x,
                                                                  prev_y, curr_x, curr_y)
                before_CP = centerPoint1

            elif handType2 == "Right" and handright2 == True \
                    and (face_direction == "Forward" or before_direction == "Forward"):
                prev_x, prev_y, curr_x, curr_y = mc.ms_controller(detector, fingers2, lmList2, width, height, frameR,
                                                                  smoothening, screen_width, screen_height, img, prev_x,
                                                                  prev_y, curr_x, curr_y)

    else:
        handright1 = False
        handright2 = False
        is_len_2 = False
        action_seq = []

    cTime = time.time()

    # if cTime != pTime:
    #     fps = 1 / (cTime - pTime)
    # else:
    #     fps = 0

    pTime = cTime

    # displaying fps
    # cv2.putText(img, f'FPS:{int(fps)}', (40, 70),
    #             cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)

    # displaying final image
    cv2.imshow('WebCam', img)
    cv2.waitKey(1)