import cv2
import numpy as np
import pickle
from os import path, walk, makedirs


def labels_for_training_data():
    """Function going through directories with sample faces and return list of that samples and list of ids to them.
    Also make file with labels to specific face"""
    current_id = 0
    label_ids = dict()
    faces, faces_ids = list(), list()

    # Go through directories and find label and path to image
    for root, dirs, files in walk('data/'):
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
    if not path.exists('labels/'):
        makedirs('labels/')
    with open('labels/face-labels.pickle', 'wb') as file:
        pickle.dump(label_ids, file)

    return faces, faces_ids


def train_classifier(train_faces, train_faces_ids):
    """Function train model to recognize face with local binary pattern histogram algorithm"""
    recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
    print('Training model in progress...')
    recognizer_lbph.train(train_faces, np.array(train_faces_ids))
    print('Saving...')
    recognizer_lbph.save('trainner.yml')
    print('Model training complete!')


new_faces, new_faces_ids = labels_for_training_data()
train_classifier(new_faces, new_faces_ids)
