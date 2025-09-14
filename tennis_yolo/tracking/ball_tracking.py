from ultralytics import YOLO
import cv2 as cv
import pickle 
import pandas as pd

class BallTracking:
    def __init__(self, model_path, fps):
        self.model = YOLO(model_path)
        self.fps = fps

    #Intepolation
    def interpolate_ball(self, ball_positions, method = 'linear', order = 3):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        if method in ['spline', 'polynomial']:
            df_positions = df_positions.interpolate(method = method, order = order)
        else:
            df_positions = df_positions.interpolate(method = method)

        df_positions = df_positions.bfill()

        ball_positions = [{1: x} for x in df_positions.to_numpy().tolist()]
        return ball_positions

    #Betect the ball
    def detect_ball(self, frames, read_from_stub = False, stub_path = None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dic = self.detect_frame(frame)
            ball_detections.append(ball_dic)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.15)[0]

        ball = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball[1] = result
        
        return ball
    
    def draw_boxes(self,video_frames, src_detections, color_box = (255, 0, 0)):
        output_frames = []
        for frame, ball in zip(video_frames, src_detections):
            for track_id, box_pos in ball.items():
                x1, y1, x2, y2 = box_pos
                cv.putText(frame, f"Ball: {track_id}", (int(box_pos[0]), int(box_pos[1] -10 )), cv.FONT_HERSHEY_COMPLEX, 0.9, color_box, 2)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
            output_frames.append(frame)
        
        return output_frames