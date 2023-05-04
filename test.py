from cvlib.object_detection import YOLO
import cv2

cap=cv2.VideoCapture(0)
weights="yolov4-tiny-custom_last.weights"
config="yolov4-tiny-custom.cfg"
labels="obj.names"

yolo = YOLO(weights, config,labels)
while True:
    ret,img=cap.read()
    img=cv2.resize(img,(680,460))
    bbox, label, conf = yolo.detect_objects(img)
    if(len(label) != 0):
        print("Detect: " + str(label))
    img1=yolo.draw_bbox(img, bbox, label, conf)
    cv2.imshow("img1",img)
    if cv2.waitKey(1)&0xFF==27:
        break
