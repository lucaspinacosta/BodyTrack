
import mediapipe as mp
import cv2
import time

class poseDetector():
    def __init__(self, mode= False,complex = 1,  smooth = True,segment = False,smoothSegment = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.complex = complex
        self.smooth = smooth
        self.segment = segment
        self.smoothSegment = smoothSegment
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complex,self.smooth,self.segment,self.smoothSegment,self.detectionCon,self.trackCon)
        """static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5"""

    def findPose(self,img, draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        
        return img
                    
    def getPose(self,img):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append({'id':id,'xPos':cx,'yPos':cy})
                #print('\n',lmList)
                """if id == 4:
                    print(id,cx,cy)"""
        return lmList
            
        

def main():
    cap = cv2.VideoCapture("videos/1.mp4")
    pTime=0
    detector = poseDetector()
    while True:
        sucess,img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPose(img)
        print("\n",lmList)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        #show fps
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)

        #adjust frame size to 1080
        frame1080 = cv2.resize(img,(1440,980),interpolation=cv2.INTER_AREA)
        
        cv2.imshow("Image", frame1080)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()