import time

import cv2
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
for i in range(1, 6):
    vid_path = "videos/"+str(i)+".mp4"
    cap = cv2.VideoCapture(vid_path)
    pTime = 0
    while True:
        sucess, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        # draw detection
        if results.pose_landmarks:
            lmList = []
            mpDraw.draw_landmarks(img, results.pose_landmarks,
                                  mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                print('\n', lmList)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # show fps
        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        # Calculate new width and height for resizing
        h, w, _ = img.shape
        aspect_ratio = w / h
        new_width = 720
        new_height = int(new_width / aspect_ratio)

        # Resize frame
        frame1080 = cv2.resize(img, (new_width, new_height),
                               interpolation=cv2.INTER_AREA)

        cv2.imshow("Image", frame1080)
        cv2.waitKey(1)
