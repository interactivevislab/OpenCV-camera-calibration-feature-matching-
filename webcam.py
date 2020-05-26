import cv2

#we need these to connect to IP cam
import urllib.request
import numpy as np
URL = "http://192.168.0.100:8080/shot.jpg"
useIPcam = True


if not useIPcam:
	#this opens the first webcam found on current PC
	cap = cv2.VideoCapture(cv2.CAP_ANY)
	
	#this is a way to define resolution of webcam video, but beware of low fps and low performance
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)

	#if your webcam supports autofocus, you'll need to disable it
	cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

while True:	
	if useIPcam:
		#open URL and read current frame as binary array
		imgNp = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
		#and decode into rgb image
		img = cv2.imdecode(imgNp,1)
	else:
		#or just read from webcam using OpenCV interface
		ret, img = cap.read()
		
	#you can also resize videoframe with interpolation
	img = cv2.resize(img, None, fx = 0.5, fy = 0.5)
	#show image
	cv2.imshow('img',img)
	
	#this command freezes output for given amount of ms (1 or greater)
	#if 0 is passed, window will wait for input forever, further commands won't be executed
	if cv2.waitKey(1) >= 0:	break #if any key is pressed

if not useIPcam: cap.release()
cv2.destroyAllWindows()