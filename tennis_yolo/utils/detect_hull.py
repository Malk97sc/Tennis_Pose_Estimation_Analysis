import numpy as np
import cv2 as cv

def hull(court_kp):
    pts = np.asarray(court_kp, dtype=np.float32).reshape(-1, 2)
    pts_int = pts.astype(np.int32)
    hull = cv.convexHull(pts_int)
    return hull