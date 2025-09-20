import numpy as np
import cv2 as cv

def refine_keypoints_subpix(source, keypoints, win_size = 10, zero_zone = (-1, -1), blur_ksize = 3):
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    h, w = gray.shape

    #cornerSubPix
    win = (win_size, win_size)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 40, 1e-3)

    refined = []
    ok_mask = []
    keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 2)

    for (x, y) in keypoints:
        #verify if the new window fits in the img
        rx = min(win_size, int(min(x, w - 1 - x)))
        ry = min(win_size, int(min(y, h - 1 - y)))
        if rx < 1 or ry < 1:
            refined.append([float(x), float(y)])
            ok_mask.append(False)
            continue

        local_win = (rx, ry)
        p = np.array([[[x, y]]], dtype=np.float32)  

        #for each point we calculate the window
        cv.cornerSubPix(gray, p, local_win, zero_zone, criteria)

        px, py = float(p[0, 0, 0]), float(p[0, 0, 1])
        if abs(px - x) <= win_size + 0.5 and abs(py - y) <= win_size + 0.5:
            refined.append([px, py])
            ok_mask.append(True)
        else:
            refined.append([float(x), float(y)])
            ok_mask.append(False)

    return np.array(refined, dtype=np.float32).reshape(-1), np.array(ok_mask, dtype=bool)