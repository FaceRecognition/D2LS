import method.CNN.face_recognition as cnn
import method.LBPH as LBPH
import argparse
from random import randint


def main():
  if args["action"]=="lbph_capture":
   print (args["action"])
  elif args["action"]=="lbph_train":
   print (args["action"])
  elif args["action"]=="lbph_detect":
   print (args["action"])
  elif args["action"]=="cnn_train":
    print (args["action"])
    fr = cnn.face_recognition(
    args["detection_method"],
    args["inputvideo"],
    args["outputvideo"],
    args["display"],
    args["image"],
    args["dataset"],
    args["encodings"],)
    fr.train()
   
  elif args["action"]=="cnn_capture":
    print (args["action"])
    fr = cnn.face_recognition(
    args["detection_method"],
    args["inputvideo"],
    args["outputvideo"],
    args["display"],
    args["image"],
    args["dataset"],
    args["encodings"],)

    fr.capture()   


  elif args["action"]=="cnn_detect_img":
    print (args["action"])
    fr = cnn.face_recognition(
    args["detection_method"],
    args["inputvideo"],
    args["outputvideo"],
    args["display"],
    args["image"],
    args["dataset"],
    args["encodings"],)

    fr.detection_img() 

  elif args["action"]=="cnn_detect_vid":
    print (args["action"])
    fr = cnn.face_recognition(
    args["detection_method"],
    args["inputvideo"],
    args["outputvideo"],
    args["display"],
    args["image"],
    args["dataset"],
    args["encodings"],)

    fr.detection_vid() 

  elif args["action"]=="cnn_detect_vid_file":
    print (args["action"])
    fr = cnn.face_recognition(
    args["detection_method"],
    args["inputvideo"],
    args["outputvideo"],
    args["display"],
    args["image"],
    args["dataset"],
    args["encodings"],)

    fr.detection_vid_file() 

  else:
   print ('NO ACTION NAMED THAT try the comand "python3 D2LS"')

    
  

  



    

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-a", "--action", type=str,required=True, help="what action you want to do")
  #cnn
  ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
  ap.add_argument("-e", "--encodings", default="D2LS/generated/CNN/encodings.pickle" ,help="path to serialized db of facial encodings")
  ap.add_argument("-iv", "--inputvideo", default="D2LS/data/videos/lunch_scene.mp4", help="path to input video")
  ap.add_argument("-ov", "--outputvideo",default="D2LS/data/output/lunch_scene_output.avi", type=str, help="path to output video")
  ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
  ap.add_argument("-i", "--image", default="D2LS/data/examples/123123.png" , help="path to input image")
  ap.add_argument("-s", "--dataset", default="D2LS/data/dataset" , help="path to input directory of faces + images")
  #LBPH
  ap.add_argument("-name", "--lbph_name", default="random" + str(randint(0, 100000)) , help="")
  ap.add_argument("-c", "--lbph_cascade", default="" , help="")
  ap.add_argument("-yml", "--lbph_yml", default="" , help="")
  ap.add_argument("-l", "--lbph_labels", default="" , help="")
  ap.add_argument("-cd", "--lbph_datapath", default="" , help="")
  ap.add_argument("-sc", "--lbph_savecapture", default="" , help="")
  ap.add_argument("-sm", "--lbph_savemodel", default="" , help="")
  args = vars(ap.parse_args())   
  main()