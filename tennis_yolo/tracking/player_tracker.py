from ultralytics import YOLO
import cv2 as cv
import pickle

class PlayerTracking:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_player(self, frames, read_stub = False, stub_path = None):
        player_dt = []

        if read_stub and stub_path is not None:
            with open(stub_path, "rb") as file:
                player_dt = pickle.load(file)
            return player_dt

        for frame in frames:
            player = self.detect_frame(frame)
            player_dt.append(player)
        
        if stub_path is not None:
            with open(stub_path, "wb") as file:
                pickle.dump(player_dt, file)
        
        return player_dt

    def detect_frame(self, frame):
        results = self.model.track(frame, persist = True)[0]
        id_name = results.names

        player = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_id = box.cls.tolist()[0]
            object_name = id_name[object_id]
            if object_name == "person":
                player[track_id] = result

        return player

    def draw_boxes(self,video_frames, player_detections, color_box = (0, 0, 255)):
        output_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, box_pos in player_dict.items():
                x1, y1, x2, y2 = box_pos
                cv.putText(frame, f"Player ID: {track_id}", (int(box_pos[0]), int(box_pos[1] -10 )), cv.FONT_HERSHEY_SIMPLEX, 0.9, color_box, 2)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
            output_frames.append(frame)
        
        return output_frames