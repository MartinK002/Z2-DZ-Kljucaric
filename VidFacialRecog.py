#za komentare na kodu vidi ImgFacialRecog.py, ovo je isti program, ali podešen da uzima frameove iz webkamere
#ili da učita neki video u pics folderu. Ja nemam webkameru, niti laptop, međutim kod bi trebao raditi za real time detekciju (barem radi za videe)

import cv2
import matplotlib.pyplot as plt

def faceBox(faceNet, frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 1)
    return frame, bboxs

faceProto = "modeli/opencv_face_detector.pbtxt"
faceModel = "modeli/opencv_face_detector_uint8.pb"

ageProto = "modeli/age_deploy.prototxt"
ageModel = "modeli/age_net.caffemodel"

genderProto = "modeli/gender_deploy.prototxt"
genderModel = "modeli/gender_net.caffemodel"


faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

#paljenje webkamere
#video = cv2.VideoCapture(0)
#paljenje videa
video = cv2.VideoCapture('pics/video1.mp4')
#petlja uzima frameove i puca ih u faceBox funkciju, ostatak postupka je identičan kao i za neku učitanu sliku
while video.isOpened():
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]


        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{}, {}".format(gender,age)
        cv2.putText(frame, label, (bbox[2]+5, bbox[3]), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 2)
    cv2.imshow("Age-Gender", frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()