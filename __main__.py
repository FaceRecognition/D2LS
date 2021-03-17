
import method.CNN.encode_faces as encode_faces
import method.CNN.recognize_faces_image as recognize_image
import method.CNN.recognize_faces_video as recognize_video
import method.CNN.recognize_faces_video_file as recognize_video_file


import method.LBPH.face_training as LBPH

import cv2
import numpy as np
import pickle
from os import path, walk, makedirs




def main():
    new_faces, new_faces_ids =  LBPH.labels_for_training_data()
    LBPH.train_classifier(new_faces, new_faces_ids)






    

if __name__ == '__main__':
    main()
   