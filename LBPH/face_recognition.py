import cv2
import pickle

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
recognizer_lbph.read("trainner.yml")

labels = {}
with open('labels/face-labels.pickle', 'rb') as file:
    org_labels = pickle.load(file)
    labels = {v: k for k, v in org_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    retval, frame = cap.read()
    # Face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap_face = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in cap_face:
        roi_gray = gray_frame[y:y + h, x:x + h]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recognition based on trained model
        id_, confidence = recognizer_lbph.predict(roi_gray)
        confidence = int(100 * (1 - (confidence / 300)))
        if confidence > 75:
            name = labels[id_]
            cv2.putText(frame, str(name) + ' ' + str(confidence) + '%', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'unknown', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
