################################################################################################
# Realtime Object Detection using YOLO and using OpenCV                                        #
# Aid Script for the Blind Detection Project                                                   #
# Developed by - Kunal Gupta, Chhavi Aggarwal and Md. Amaan Nehru                              # 
# Boilerplate code courtesy of Arun Poswammy                                                   #
# Fire a terminal and browse to the directory you are viewing this script in                   #
# Then type the following to run the script                                                    #
# python final_script.py --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt     #
################################################################################################

import cv2
import argparse
import numpy as np
from imutils.video import FPS
import time
from mss import mss
from PIL import Image
import pygame
from pygame import mixer 
import os


pygame.init()
## Adding arguments for running the framework 
ap  = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True
                , help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def playvoice(class_name, path):
    print "Class Name is:{}".format(class_name)
    if class_name == "person":
        mixer.music.load(path + 'person.mp3')
        mixer.music.play()
    elif class_name == "chair":
        mixer.music.load(path + 'chair.mp3')
        mixer.music.play()
    elif class_name == "dining table":
        mixer.music.load(path + 'diningtable.mp3')
        mixer.music.play()
    elif class_name == "laptop":
        mixer.music.load(path + 'laptop.mp3')
        mixer.music.play()
    elif class_name == "car":
        mixer.music.load(path + 'car.mp3')
        mixer.music.play()
    else:
        mixer.music.load(path + 'default.mp3')
        mixer.music.play()



classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)
fps = FPS().start()
print "Starting YOLO Object Detection Script..."
mon = {'top':270 , 'left': 120 , 'width': 600 , 'height': 420 }
sct = mss()
st = time.time()
voiceOK = True
path = os.getcwd()+'/Audio Commands/'
mixer.init()
# play the intro
mixer.music.load(path + 'intro.mp3')
mixer.music.play()
while True:
    sct.get_pixels(mon)
    image = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    image = np.array(image)
    Width, Height = image.shape[1], image.shape[0]
    scale = 0.000392
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices)==0:
        cv2.imshow("Object Detection", image)
    else:
        max_confidence = 0.000
        id = 0
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box[0], box[1],box[2], box[3]
            if max_confidence < confidences[i]:
                max_confidence = confidences[i]
                id = i
            draw_prediction(image, class_ids[i], confidences[i], int(x), int(y), int(x+w), int(y+h))
            cv2.imshow("Object Detection", image)
        if voiceOK == True:
            playvoice(classes[class_ids[id]], path)
            voiceOK = False
            st = time.time()
        
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    et = time.time()
    if et - st >= 4.00:
        voiceOK = True

fps.stop()
# play the extro
mixer.music.load(path + 'extro.mp3')
mixer.music.play()
pygame.event.wait()
print "Average FPS:"+str(fps.fps())
cv2.destroyAllWindows()         
