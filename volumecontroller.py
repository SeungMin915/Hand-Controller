from lib import *
        
##볼륨
def vol_controller(lmList, volume, volRange, img):
    minVol = volRange[0]
    maxVol = volRange[1]
    
    x1, y1 = lmList[4][0], lmList[4][1]
    x2, y2 = lmList[8][0], lmList[8][1]
    x3, y3 = lmList[12][0], lmList[12][1]
    x4, y4 = lmList[16][0], lmList[16][1]
    x5, y5 = lmList[20][0], lmList[20][1]
    p1, p2 = lmList[9][0], lmList[9][1]
    p3, p4 = lmList[0][0], lmList[0][1]

    cx, cy = (p1 + p3) // 2, (p2 + p4) // 2

    h2 = math.hypot(p3 - p1, p4 - p2)
    c2 = round(math.acos((p3 - p1)/h2) * (180 / math.pi))

    resultVolume=round(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'{int(resultVolume)}%', (40, 450),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 3)

    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

    try:
        if (c2<48):
            volume.SetMasterVolumeLevel(-64.4974136352539, None)
        elif (48<=c2<60):
            volume.SetMasterVolumeLevel(-33.33053970336914, None)
        elif (60<=c2<72):
            volume.SetMasterVolumeLevel(-23.61930274963379, None)
        elif (72<=c2<84):
            volume.SetMasterVolumeLevel(-17.741300582885742, None)
        elif (84<=c2<96):
            volume.SetMasterVolumeLevel(-13.513073921203613, None)
        elif (96<=c2<108):
            volume.SetMasterVolumeLevel(-10.20866584777832, None)
        elif (108<=c2<120):
            volume.SetMasterVolumeLevel(-7.746397495269775, None)
        elif (120<=c2<132):
            volume.SetMasterVolumeLevel(-5.409796714782715, None)
        elif (132<=c2<144):
            volume.SetMasterVolumeLevel(-3.384982109069824, None)
        elif (144<=c2<156):
            volume.SetMasterVolumeLevel(-1.5984597206115723, None)
        elif (c2 >= 156):
                volume.SetMasterVolumeLevel(0.0, None)
    except :
        pass