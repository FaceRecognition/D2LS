import cv2
from os import path, makedirs

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(cap_frame):
    """Function detect face and return region of interest"""
    gray_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2GRAY)
    cap_face = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in cap_face:
        roi = cap_frame[y:y + h, x:x + w]
        return roi


print('Enter your name: ', end='')
name = input()
print('Enter your surname: ', end='')
surname = input()
directory = name+'-'+surname

# If not exist create directory for new user samples
if not path.exists('data/'+str(directory.lower())):
    makedirs('data/'+str(directory.lower()))

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # If face was detected save region of interest in user directory
        file_name_path = 'data/'+str(directory.lower())+'/'+str(count)+'.jpg'
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
