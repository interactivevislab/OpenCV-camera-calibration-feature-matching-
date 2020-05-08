import cv2
import numpy as np
import urllib.request
import os

URL = "http://192.168.0.101:8080/shot.jpg"
useIPcam = False
cap = None
if not useIPcam: cap = cv2.VideoCapture(cv2.CAP_DSHOW)

def getFrame():	
	if useIPcam:
		imgNp = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
		frame = cv2.imdecode(imgNp,-1)
	else: 
		ret, frame = cap.read()
	
	return frame
	
	
def drawPatch(frame, patch, homography):
	h, w, n = frame.shape
	warpedPatch = cv2.warpPerspective(patch, homography, (w, h))
	#cv2.imshow('wp', warpedPatch)
	mask = np.zeros((h,w,1), np.uint8)
	h,w,_ = patch.shape
	pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
	# project corners into frame
	dst = cv2.perspectiveTransform(pts, homography)
	
	#fill patch area on mask
	mask = cv2.fillPoly(mask, [np.int32(dst)], 255)

	src1final = cv2.copyTo(frame, cv2.bitwise_not(mask)) #copy original frame in every point except where patch will be drawn
	src2final = cv2.copyTo(warpedPatch, mask) #fill the gap with warped patch image
	final = src1final+src2final
	cv2.imshow('patchdraw', final)
	
	return frame


if __name__ == '__main__':
	#homography warping transformation - 3x3 matrix
	homography = None 
	#matrix of camera parameters
	camera_parameters = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
	
	#ORB keypoint detector
	orb = cv2.ORB_create()
	
	# create BFMatcher object based on hamming distance  
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	#or flann-based
	FLANN_INDEX_LSH = 6
	index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	
	# load the reference image that will be recognized in the video stream
	dir_name = os.getcwd()
	model = cv2.imread(os.path.join(dir_name, 'reference/card (3).jpg'), 0)
	patch = cv2.imread(os.path.join(dir_name, 'reference/pic.jpg'), 1)
	h,w = model.shape[:2]
	#note: h,w -> w,h to resize 
	patch = cv2.resize(patch, (w,h))
	
	# Compute model keypoints and its descriptors - they're fixed and will be used later
	kp_model, des_model = orb.detectAndCompute(model, None)
	
	# Minimum number of matches that have to be found
	# to consider the recognition valid
	MIN_MATCHES = 10  

	while True:
		
		frame = getFrame()
		
		#frame descriptors, matches and filtered matches
		des_frame = []
		matches = []
		good_matches = []
		
		# find the features of the frame
		kp_frame, des_frame = orb.detectAndCompute(frame, None)

		if len(des_frame) > MIN_MATCHES:
			# match frame descriptors with model descriptors
			matches = flann.knnMatch(des_model, des_frame, k=2)
		if len(matches) > MIN_MATCHES:
			
			#filter outliers
			for m_n in matches:
				if len(m_n) != 2: 
					continue
				(m,n) = m_n
				if m.distance < 0.7*n.distance:
					good_matches.append(m)
			
			# compute Homography if enough matches are found
			if len(good_matches) > MIN_MATCHES:
				# differenciate between source points and destination points
				src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
				dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
				# compute Homography
				homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
				
				if homography is not None:
					frame = drawPatch(frame, patch, homography)
					
					#draw rectangle frame of detected marker
					h, w = model.shape
					pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
					# project corners into frame
					dst = cv2.perspectiveTransform(pts, homography)
					frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
				
				#endif homography is not None	
			#endif len(good_matches) > MIN_MATCHES
		#endif len(matches) > MIN_MATCHES
		
		#draw detected good_matches matches (even if matches not found)
		frame = cv2.drawMatches(model, kp_model, frame, kp_frame, good_matches[:], 0, flags=2)
				
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) > 0 :
			break #breaks main loop if any key pressed
	
	#end while True
	
	#release resources before exit
	cap.release()
	cv2.destroyAllWindows()

#EOF