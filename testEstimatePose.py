import pickle
import numpy as np
import cv2
import math

# Lecture
with open("data.pkl", "rb") as f:
    l_kps_ref, l_kps_cur = pickle.load(f)

intrinsic =  (640, 480, 319.99999999999994, 319.99999999999994, 320.0, 240.0)
width, height, fx, fy, cx, cy = intrinsic

kRansacProb = 0.999  # (originally 0.999)
kMinAveragePixelShiftForMotionEstimation = 1.5  # if the average pixel shift is below this threshold, motion is considered to be small enough to be ignored
kRansacThresholdNormalized = ( 0.0004)  # metric threshold used for normalized image coordinates (originally 0.0003)
absolute_scale = 1.0

Kinv = np.array(
    [[1.0 / fx, 0.0, -cx / fx], [0.0, 1.0 / fy, -cy / fy], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

def add_ones(x):
    if len(x.shape) == 1:
        return np.array([x[0], x[1], 1])
        # return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def unproject_points( uvs):  # numba version
    uvs = uvs.astype(np.float64)
    return np.dot(Kinv, add_ones(uvs).T).T[:, 0:2]

# [4x4] homogeneous T from [3x3] R and [3x1] t
# @jit(nopython=True)
def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret


# [4x4] homogeneous inverse T^-1 in SE(3) from T represented with [3x3] R and [3x1] t
# @jit(nopython=True)
def inv_poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R.T
    ret[:3, 3] = -R.T @ np.ascontiguousarray(t)
    return ret


#verstion chatGPT
def is_rotation_matrix(R, tol=1e-8):
    # Vérifie taille 3x3
    if R.shape != (3, 3):
        return False

    # Produit scalaire entre lignes → identité implicite
    r0, r1, r2 = R
    dot01 = np.dot(r0, r1)
    dot02 = np.dot(r0, r2)
    dot12 = np.dot(r1, r2)
    
    # Normes proches de 1
    n0 = np.dot(r0, r0)
    n1 = np.dot(r1, r1)
    n2 = np.dot(r2, r2)
    
    # Conditions d'orthogonalité et de normalisation
    if not (abs(dot01) < tol and abs(dot02) < tol and abs(dot12) < tol):
        return False
    if not (abs(n0 - 1.0) < tol and abs(n1 - 1.0) < tol and abs(n2 - 1.0) < tol):
        return False

    # Déterminant proche de 1
    det = (
        R[0,0]*(R[1,1]*R[2,2] - R[1,2]*R[2,1])
        - R[0,1]*(R[1,0]*R[2,2] - R[1,2]*R[2,0])
        + R[0,2]*(R[1,0]*R[2,1] - R[1,1]*R[2,0])
    )

    return abs(det - 1.0) < tol

# --- CONVERSION MATRICE → ANGLES EULER (convention ZYX : yaw, pitch, roll) ---
def rotm_to_euler_zyx(R):
    if abs(R[2, 0]) < 1 - 1e-6:
        pitch = -math.asin(R[2, 0])
        roll = math.atan2(R[2, 1] / math.cos(pitch), R[2, 2] / math.cos(pitch))
        yaw = math.atan2(R[1, 0] / math.cos(pitch), R[0, 0] / math.cos(pitch))
    else:
        # Cas de gimbal lock
        roll = 0.0
        if R[2, 0] <= -1:
            pitch = math.pi / 2
            yaw = math.atan2(R[0, 1], R[0, 2])
        else:
            pitch = -math.pi / 2
            yaw = math.atan2(-R[0, 1], -R[0, 2])
    return roll, pitch, yaw


cur_R = np.eye(3, 3)  # current rotation Rwc
cur_t = np.zeros((3, 1))  # current translation twc

init_history = True
traj3d_est = []
poses= []

for i,(c,r) in enumerate(zip( l_kps_cur,l_kps_ref)):
    # print(i)
    kpn_ref=unproject_points(r)
    kpn_cur =unproject_points(c)

    # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
    E, mask_match = cv2.findEssentialMat(
        kpn_cur,
        kpn_ref,
        focal=1,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=kRansacProb,
        threshold=kRansacThresholdNormalized,
    )
    
    pose_estimation_inliers, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0.0, 0.0))
    
    cur_R = cur_R.dot(R)
    if not is_rotation_matrix(cur_R):
        print(f"Correcting rotation matrix: {cur_R}")
        # cur_R = self.closest_rotation_matrix(self.cur_R)
    cur_t = cur_t + absolute_scale * cur_R @ t
    # print(cur_t)

    if init_history : 
       t0_est = np.array([cur_t[0], cur_t[1], cur_t[2]])  # starting translation
       T0_inv_est = inv_poseRt(cur_R, cur_t.ravel())
       init_history = False

     # p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
    cur_T = T0_inv_est @ poseRt(cur_R, cur_t.ravel())
    p = cur_T[:3, 3].ravel()
    traj3d_est.append(p)

    pose = poseRt(cur_R, np.array(p).ravel())
    poses.append(pose)
    # print(pose)

    #### affichage position
    M = poses[-1]
    t = M[:3, 3]
    roll, pitch, yaw = rotm_to_euler_zyx(M)
    print(f"{i} : x={t[0]:.6f}, y={t[1]:.6f}, z={t[2]:.6f} | Roll  : {math.degrees(roll):.3f}°, Pitch : {math.degrees(pitch):.3f}°,Yaw   : {math.degrees(yaw):.3f}°")
    #####
