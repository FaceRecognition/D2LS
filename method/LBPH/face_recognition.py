import cv2
import pickle
from os import path, makedirs
from os import path, walk, makedirs
import numpy as np


class face_recognition:
    def __init__(self,  detection_method, 
                        inputvideo,
                        outputvideo,
                        display,
                        image,
                        dataset,
                        encodings,):
        self.detection_method = detection_method
        self.inputvideo = inputvideo
        self.outputvideo = outputvideo
        self.display = display
        self.image = image
        self.dataset = dataset
        self.encodings = encodings

    def capture(self):

       
        print('Enter your name: ', end='')
        name = input()
        print('Enter your surname: ', end='')
        surname = input()
        directory = name+'-'+surname

        # If not exist create directory for new user samples
        if not path.exists('D2LS/method/LBPH/data/'+str(directory.lower())):
            makedirs('D2LS/method/LBPH/data/'+str(directory.lower()))

        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            if self.face_extractor(frame) is not None:
                face = cv2.resize(self.face_extractor(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # If face was detected save region of interest in user directory
                file_name_path = 'D2LS/method/LBPH/data/'+str(directory.lower())+'/'+str(count)+'.jpg'
                cv2.imwrite(file_name_path, face)
                count += 1

                # Print number of already saved samples
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, 200, 2)
                cv2.imshow('Collecting samples', face)
            else:
                print('Face not found')

            if cv2.waitKey(30) & 0xFF == ord('q') or count == 150:
                break

        cap.release()
        cv2.destroyAllWindows()
        print('Collecting samples complete!')

        
    def detection(self):

        face_classifier = cv2.CascadeClassifier('D2LS/method/LBPH/haarcascade_frontalface_default.xml')
        recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
        recognizer_lbph.read("D2LS/method/LBPH/trainner.yml")

        labels = {}
        with open('D2LS/method/LBPH/labels/face-labels.pickle', 'rb') as file:
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
    def face_extractor(self ,cap_frame):
        face_classifier = cv2.CascadeClassifier('D2LS/method/LBPH/haarcascade_frontalface_default.xml')

        gray_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2GRAY)
        cap_face = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in cap_face:
            roi = cap_frame[y:y + h, x:x + w]
            return roi



    def labels_for_training_data(self ,):
        """Function going through directories with sample faces and return list of that samples and list of ids to them.
        Also make file with labels to specific face"""
        current_id = 0
        label_ids = dict()
        faces, faces_ids = list(), list()

        # Go through directories and find label and path to image
        for root, dirs, files in walk('D2LS/method/LBPH/data/'):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    img_path = path.join(root, file)
                    label = path.basename(root).replace(' ', '-').lower()
                    if label not in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]

                    test_img = cv2.imread(img_path)
                    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                    if test_img is None:
                        print('Image not loaded properly')
                        continue

                    faces.append(test_img)
                    faces_ids.append(id_)

        # Make directory with labels doesn't exist make directory and file with labels
        if not path.exists('D2LS/method/LBPH/labels/'):
            makedirs('D2LS/method/LBPH/labels/')
        with open('D2LS/method/LBPH/labels/face-labels.pickle', 'wb') as file:
            pickle.dump(label_ids, file)

        return faces, faces_ids


    def train(self ,train_faces, train_faces_ids):
        """Function train model to recognize face with local binary pattern histogram algorithm"""
        recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
        print('Training model in progress...')
        recognizer_lbph.train(train_faces, np.array(train_faces_ids))
        print('Saving...')
        recognizer_lbph.save('D2LS/method/LBPH/trainner.yml')
        print('Model training complete!')
    



