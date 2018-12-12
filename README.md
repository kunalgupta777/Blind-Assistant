# Blind Assistant using YOLO and OpenCV
This repository contains the code for the Blind Assistant.
Under development.
## Requirements
 - OpenCV 3.4.x
 - Python 2.7.x
 - PIL
 - MSS
 - imutils
 - IP Webcam Android Application ( find it on Play Store )
 - YOLO v3 weights file 
 ## Instructions
 Download YOLOv3 weights (~240 MB) by adding
 ```
 :~$ wget https://pjreddie.com/media/files/yolov3.weights
 ```
 Clone this repository and run the script by adding
 ```
 :~$ python final_script.py --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
 ```
 The script takes continuous screenshots of the topleft corner of the display port and uses that to detect objects.
 Use IP Webcam to relay live video feed and place the browser window at the appropriate location.
 
 ## To Do
  - [ ] Auotmate the whole process of window minimisation
  - [ ] Add tiny yolo support
  - [ ] Add more Audio Commands
  - [ ] Add GPU suppport to run YOLO ( using CUDA )
  - [ ] Port the ecosystem to Android Device 
