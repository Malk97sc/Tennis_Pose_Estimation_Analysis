import cv2 as cv
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pickle

class PlayerPose:
    def __init__(self, model_path, kp_radius = 3, conf_threshold = 0.15, pad = 20, imgsz = 640):
        self.model = YOLO(model_path)
        self.skeleton = [
            (5, 6), #Shoulders
            (5, 7), (7, 9), #Left Arm    
            (6, 8), (8, 10), #Right Arm
            (11, 12), #Hips
            (11, 13), (13, 15), #Left Leg
            (12, 14), (14, 16), #Right Leg
            (5, 11), (6, 12) #Torso (shoulders to hips)
        ]
        self.radius = kp_radius
        self.conf = conf_threshold
        self.pad = pad
        self.imgsz = imgsz
    
    def _crop_with_padding(self, frame, bbox, pad = 20):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1i = int(max(0, np.floor(x1 - pad)))
        y1i = int(max(0, np.floor(y1 - pad)))
        x2i = int(min(w - 1, np.ceil(x2 + pad)))
        y2i = int(min(h - 1, np.ceil(y2 + pad)))
        crop = frame[y1i:y2i+1, x1i:x2i+1].copy()
        return crop, (x1i, y1i, x2i, y2i)

    def _map_keypoints(self, kps, crop_box, crop_shape):
        x1, y1, x2, y2 = crop_box
        ch, cw = crop_shape
        mapped = []
        for (x, y, conf) in kps:
            if 0 <= x <= 1 and 0 <= y <= 1:  
                xf = float(x) * cw
                yf = float(y) * ch
            else:
                xf, yf = float(x), float(y)
            mapped.append([xf + x1, yf + y1, float(conf)])
        return np.array(mapped, dtype=float)
    
    def _extract_keypoints(self, res):
        if hasattr(res, "keypoints") and res.keypoints is not None:
            try:
                return [np.hstack([np.array(k.xy).reshape(-1,2),
                                   np.array(k.conf).reshape(-1,1)]) for k in res.keypoints]
            except Exception:
                pass
        if hasattr(res, "boxes") and hasattr(res.boxes, "keypoints"):
            raw = res.boxes.keypoints
            if isinstance(raw, np.ndarray):
                n, total = raw.shape
                kp_count = total // 3
                return [raw[i].reshape(kp_count, 3) for i in range(n)]
        return []
    
    def calc_estimate_pose(self, video, player_dt):
        player_pose_dt = []
        frame_num = 0
        for frame_idx, frame in enumerate(video):
            frame_bboxes = player_dt[frame_idx] if frame_idx < len(player_dt) else {}
            crops, crop_boxes, ids = [], [], []
            for pid, bbox in frame_bboxes.items():
                crop, box = self._crop_with_padding(frame, bbox)
                if crop.size == 0:
                    continue
                crops.append(crop)
                crop_boxes.append(box)
                ids.append(pid)

            frame_pose_dict = {}
            if len(crops) > 0:
                results = self.model.predict(crops, imgsz=self.imgsz,
                                             conf=self.conf, verbose=False)
                for i, res in enumerate(results):
                    kp_instances = self._extract_keypoints(res)
                    if len(kp_instances) == 0:
                        continue
                    best_kp = max(kp_instances, key=lambda arr: arr[:,2].mean())
                    mapped = self._map_keypoints(best_kp, crop_boxes[i], crops[i].shape[:2])
                    frame_pose_dict[ids[i]] = mapped
            player_pose_dt.append(frame_pose_dict)
            print(f"Frame: {frame_num}")
            frame_num += 1

        return player_pose_dt
    
    def estimate_pose(self, video, player_dt, read_stub = False, stub_path = None):
        if read_stub and stub_path is not None:
            with open(stub_path, "rb") as file:
                player_pose_dt = pickle.load(file)
            return player_pose_dt

        player_pose_dt = self.calc_estimate_pose(video, player_dt)

        if stub_path is not None:
            with open(stub_path, "wb") as file:
                pickle.dump(player_pose_dt, file)

        return player_pose_dt
    
    def draw_pose(self, video, player_pose_dt, color=(255, 255, 0)):
        out_video = []
        skip_points = 4 #nose, left eye, right eye and ears

        for i, frame in enumerate(video):
            pose_dict = player_pose_dt[i] if i < len(player_pose_dt) else {}
            for pid, kps in pose_dict.items():
                #points of yolo
                for idx, (x, y, conf) in enumerate(kps):
                    if conf >= self.conf and idx > skip_points:
                        cv.circle(frame, (int(x), int(y)), self.radius, color, -1)
                #skeleton
                for a, b in self.skeleton:
                    if a <= skip_points or b <= skip_points:
                        continue 
                    if a < len(kps) and b < len(kps):
                        xa, ya, ca = kps[a]
                        xb, yb, cb = kps[b]
                        if ca >= self.conf and cb >= self.conf:
                            cv.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
            out_video.append(frame)
        return out_video


