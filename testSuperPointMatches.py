
import cv2
import torch
import numpy as np
import tf2_ros
import tf_conversions
from pathlib import Path

from superpoint.detector import SuperPointDetector

intrinsic =  (640, 480, 319.99999999999994, 319.99999999999994, 320.0, 240.0)
width, height, fx, fy, cx, cy = intrinsic
K = np.array([[fx   ,0.0, cx ],
                [0.0  ,fy , cy ],
                [0.0  ,0.0, 1.0]])  


def image_callback_cv2match(img):

    global prev_img
    global prev_kp
    global prev_desc,frame_id,pose
    global K

    kp, desc = detector.traitement(img,True)
    desc = desc.T
    kp = kp.T[:,:2]
    imgR = img
    if prev_img is not None:
        # Matching descripteurs (brut-force L2)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(prev_desc, desc)
        #matches = bf.knnMatch(des1, des2, k=2) 
        
        matches = sorted(matches, key=lambda x: m.queryIdx)
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([prev_kp[m.queryIdx] for m in matches])
        pts2 = np.float32([kp[m.trainIdx] for m in matches])

        # Filtrage RANSAC pour obtenir Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            print("Pas de matrice essentielle trouvée.")
            return

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        scaler=(img.shape[1]/detector.w,img.shape[0]/detector.h)
        kps1 = [cv2.KeyPoint(float(p[0]* scaler[0]), float(p[1] * scaler[1]), 1) for p in pts1]
        kps2 = [cv2.KeyPoint(float(p[0]* scaler[0]), float(p[1] * scaler[1]), 1) for p in pts2]
        # matches_cv = [cv2.DMatch(i, i, 0) for i in range(len(pts1))]
        imgR = cv2.drawMatches(prev_img, kps1, img, kps2, matches, None, flags=2)

        # Mise à jour de la pose globale
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        pose = pose @ np.linalg.inv(T)
        print(pose[:3, 3])

    prev_img = img
    prev_kp = kp
    prev_desc = desc
    frame_id += 1
    
    return imgR
        
    
def image_callback_superpointmatch(img):

    global prev_img
    global prev_kp
    global prev_desc,frame_id,pose
    global K

    kp, desc,matches = detector.traitement(img,matches=True)

    desc = desc.T
    kp = kp.T[:,:2]
    if prev_img is not None:
        #matches = bf.knnMatch(des1, des2, k=2) 
        #matches_bf = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32(prev_kp[matches[0].astype(int)])
        pts2 = np.float32(kp[matches[1].astype(int)])

        # Filtrage RANSAC pour obtenir Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            print("Pas de matrice essentielle trouvée.")
            return

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        scaler=(img.shape[1]/detector.w,img.shape[0]/detector.h)
        kps1 = [cv2.KeyPoint(float(p[0]* scaler[0]), float(p[1] * scaler[1]), 1) for p in pts1]
        kps2 = [cv2.KeyPoint(float(p[0]* scaler[0]), float(p[1] * scaler[1]), 1) for p in pts2]
        
        matches_cv = [cv2.DMatch(i, i, 0) for i in range(len(pts1))]

        img3 = cv2.drawMatches(prev_img, kps1, img, kps2, matches_cv, None, flags=2)

        # Mise à jour de la pose globale
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        pose = pose @ np.linalg.inv(T)
        
        return img3

    prev_img = img
    prev_kp = kp
    prev_desc = desc
    frame_id += 1    

def image_callback_cv2ORB(img):

    global prev_img
    global prev_kp
    global prev_desc,frame_id,pose
    global K

    # Initiate SIFT detector
    extract = cv2.SIFT_create()
    # extract = cv2.ORB_create()
    kp, desc = extract.detectAndCompute(img, None)
    if prev_img is not None:
        # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # matches = matcher.match(prev_desc, desc)
        # img3 = cv2.drawMatches(prev_img, prev_kp, img, kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        matcher = cv2.FlannBasedMatcher(index_params,search_params)
        matches = matcher.knnMatch(prev_desc,desc,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        
        img3 = cv2.drawMatchesKnn(img1,prev_kp,img2,kp,matches,None,**draw_params)

        return img3

    prev_img = img
    prev_kp = kp
    prev_desc = desc
    frame_id += 1    


if __name__ == '__main__':


    OUTPUT_WIDTH = 740
    OUTPUT_HEIGHT = 555

    img1 = cv2.imread("/home/cy/dev-python/workspace/autreProjet/img/i1.png")
    img2 = cv2.imread("/home/cy/dev-python/workspace/autreProjet/img/i2.png")

    prev_img = None
    prev_kp = None
    prev_desc = None
    pose = np.eye(4)
    frame_id = 0

    #match par CV2
    detector = SuperPointDetector()
    detector.initialisation()
    
    img = img1
    while True:
        i = image_callback_cv2match(img)
        cv2.imshow("Webcam", i)
        key = cv2.waitKey(1)
        if key != -1:
            if img != img2:
                img = img2
            else: 
                break  
    
    #match par supervpoint tracker
    # prev_img = None
    # prev_kp = None
    # prev_desc = None
    # pose = np.eye(4)
    # frame_id = 0
    # detector = SuperPointDetector()
    # detector.initialisation()
    # image_callback_superpointmatch(img1)
    # i2 = image_callback_superpointmatch(img2)
    
    # image_callback_cv2ORB(img1)
    # i = image_callback_cv2ORB(img2)
    

    # i = np.vstack((i1, i2))
    # while True:
    #     cv2.imshow("Webcam", i)
    #     key = cv2.waitKey(1)
    #     if key != -1:
    #         break  
    


    cv2.destroyAllWindows()