import mediapipe as mp
import cv2
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture("videos/4.mp4")
pTime=0
while True:
    sucess,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #draw detection 
    if results.pose_landmarks:
        lmList=[]
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            #print(id, lm)
            h, w, c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            lmList.append([id,cx,cy])
            print('\n',lmList)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    #show fps
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)

    #adjust frame size to 1080
    frame1080 = cv2.resize(img,(1020,780),interpolation=cv2.INTER_AREA)
    

        
    cv2.imshow("Image", frame1080)
    cv2.waitKey(1)