import cv2
import matplotlib.pyplot as plt

#funkcija koja će čitati lica sa slike
def faceBox(faceNet, frame):
    frameWidth = frame.shape[1]     #definiranje veličine framea
    frameHeight = frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104, 117, 123], swapRB=False)      #pretvaranje slike u oblik koji neuralna mreža prihvaća (blob)
    faceNet.setInput(blob)
    detection = faceNet.forward()           #detekcija lica
    bboxs = []
    for i in range(detection.shape[2]):     #izlaz je detekcijska matrica, tj. ugnježđena lista raznih vrijednosti. na 2. indeksu je confidence value detekcija
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:                #ako je confidence manji od 70% odbacuje se detekcija, inace se definiraju koordinate bounding boxeva
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])     #spremamo boxeve u listu bboxs
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 1)     #crtamo rectangle na sliku i vraćamo frame uz bounding boxes
    return frame, bboxs


#učtavanje modela za detekciju lica
faceProto = "modeli/opencv_face_detector.pbtxt"
faceModel = "modeli/opencv_face_detector_uint8.pb"

#učitavanje modela za određvianje dobi
ageProto = "modeli/age_deploy.prototxt"
ageModel = "modeli/age_net.caffemodel"

#učitavanje modela za određivanje spola
genderProto = "modeli/gender_deploy.prototxt"
genderModel = "modeli/gender_net.caffemodel"

#učitavanje neuralnih mreža za detekciju lica, spola i dobi
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

#podatci za model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


#učivanje slike
frame = cv2.imread('pics/Evil_Jerma.png')

#dobivanje bounding boxesa
frame, bboxs = faceBox(faceNet, frame)
for bbox in bboxs:      #svaki bounding box sadrži detektirano lice, svako lice se pretvara u oblik iskoristiv mreži (blob) i prosljeđuje se mrežama za spol i dob
    face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]


    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]
    #postavljanje labela na sliku
    label = "{}, {}".format(gender,age)         
    cv2.putText(frame, label, (bbox[2]+5, bbox[3]), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 2)
cv2.imshow("Age-Gender", frame)
cv2.waitKey(0)