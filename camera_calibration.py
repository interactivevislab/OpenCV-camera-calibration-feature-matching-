import numpy as np
import cv2
import urllib.request 

ESC = 27
SPACE = 32
CHECKERBOARD_SIZE = (4, 6)

URL = "http://192.168.0.100:8080/shot.jpg"
useIPcam = True

cap = None
if not useIPcam:
	cap = cv2.VideoCapture(cv2.CAP_DSHOW)

def getFrame():	
	if useIPcam:
		imgNp = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
		frame = cv2.imdecode(imgNp,-1)
	else: 
		ret, frame = cap.read()
	
	return frame

def capture(mtx, newcameramtx, dist, roi):

	# Capture frame-by-frame
	while cv2.waitKey(1) == -1:

		frame = getFrame()
		# undistort
		undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
		# crop the image
		x, y, w, h = roi
		undistorted = undistorted[y:y+h, x:x+w]

		# Display the resulting frame
		cv2.imshow('frame', frame)
		cv2.imshow('undistorted frame', undistorted)


def reproj_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
	mean_error = 0
	for i in range(len(objpoints)):
		imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
		mean_error += error

	return mean_error / len(objpoints)


def calibrate_checkerboard(chess_size=CHECKERBOARD_SIZE):
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
	objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)
	
	objpoints = []
	imgpoints = []
	
	#capture points every 10th time they're detected
	cooldown = 10
	captured = 0
	active = False
	
	while True:
		key = cv2.waitKey(1)
		if key == ESC:
			break

		#switch capturing process
		if key == SPACE:
			active = not active
			if active:	print("Recording")
			else:       print("Not recording")

		#read next video frame
		frame = getFrame()

		cv2.imshow('frame', frame)

		#if not recording
		if not active: continue

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, chess_size, None)

		cv2.drawChessboardCorners(frame, chess_size, corners, ret)
		cv2.imshow('frame', frame)
		
		#if not detected
		if not ret : continue
		
		print("  Checkerboard of size {} found".format(chess_size))
		
		#capture only 10th detected corners, otherwise continue
		cooldown -= 1
		if cooldown > 0: continue
		
		cooldown = 10
		
		cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
		objpoints.append(objp)
		imgpoints.append(corners)

		captured += 1
		print("Captured {}".format(captured))
		if captured >= 100: break

	print("Calibration")
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	print("reproj error: ", reproj_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist))

	#fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, (w, h), apertureWidth, apertureHeight)

	return ret, mtx, dist, rvecs, tvecs, frame.shape


if __name__ == "__main__":	

	print("Calibrating")
	ret, mtx, dist, rvecs, tvecs, (h, w) = calibrate_checkerboard()

	#print("Refining camera matrix")
	#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

	print("Saving results")
	#np.savetxt("calibration/new_camera_matrix.txt", newcameramtx, fmt="%f")
	np.savetxt("calibration/camera_matrix.txt", mtx, fmt="%f")
	np.savetxt("calibration/distortion.txt", dist, fmt="%f")

	#print("Starting capture")
	#capture(mtx, newcameramtx, dist, roi)

	print("Done")
	cap.release()
	cv2.destroyAllWindows()



