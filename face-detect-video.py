from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True,help="Caffe modulu yolu, prototxt dosyası için")
ap.add_argument("-m","--model",required=True,help="Caffe pre-trained model yolu")
ap.add_argument("-c","--confidence",type=float, default=0.5,help="zayıf filtreleme için min olasılık")
args = vars(ap.parse_args())

print("model yukleniyor")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

print("video baslatılıyor")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,177,123))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence < args["confidence"]:
            continue
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (xBas,yBas,xSon,ySon) = box.astype("int")

        text = "{:2f}%".format(confidence*100)
        y = yBas - 10 if yBas - 10 > 10 else yBas+10
        
        cv2.rectangle(frame,(xBas,yBas),(xSon,ySon),(0,0,255),2)

        cv2.putText(frame,text,(xBas,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
    
    cv2.imshow("videostream",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
vs.stop()
