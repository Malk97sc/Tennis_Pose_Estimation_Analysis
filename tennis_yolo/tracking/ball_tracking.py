from ultralytics import YOLO
import cv2 as cv
import pickle 
import pandas as pd

class BallTracking:
    def __init__(self, model_path, fps, confidence = 0.15):
        self.model = YOLO(model_path)
        self.fps = fps
        self.conf = confidence

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
    
    #Ball hit
    def get_shot_frames(self, ball_pos, changes = 20):
        ball_pos = [x.get(1, []) for x in ball_pos]
        df_ball_pos = pd.DataFrame(ball_pos, columns = ['x1', 'y1', 'x2', 'y2'])

        df_ball_pos['ball_hit'] = 0 #the ball was hit   
        df_ball_pos['mid_y'] = (df_ball_pos['y1'] + df_ball_pos['y2']) / 2
        df_ball_pos['mid_y_rolling_mean'] = df_ball_pos['mid_y'].rolling(window = 5, min_periods = 1).mean()
        df_ball_pos['delta_y'] = df_ball_pos['mid_y_rolling_mean'].diff()

        change_frames_for_hit = changes

        for i in range(1, len(df_ball_pos) - int(change_frames_for_hit * 1.2)):
            negative_position = df_ball_pos['delta_y'].iloc[i] > 0 and df_ball_pos['delta_y'].iloc[i + 1] < 0
            positive_position = df_ball_pos['delta_y'].iloc[i] < 0 and df_ball_pos['delta_y'].iloc[i + 1] > 0

            if negative_position or positive_position:
                change_count = 0
                for change_frame in range(i + 1, i + int(change_frames_for_hit * 1.2) + 1):
                    neg_follow = df_ball_pos['delta_y'].iloc[i] > 0 and df_ball_pos['delta_y'].iloc[change_frame] < 0
                    pos_follow = df_ball_pos['delta_y'].iloc[i] < 0 and df_ball_pos['delta_y'].iloc[change_frame] > 0

                    if negative_position and neg_follow:
                        change_count += 1
                    elif positive_position and pos_follow:
                        change_count += 1

                if change_count > change_frames_for_hit - 1:
                    df_ball_pos.loc[i, 'ball_hit'] = 1 

        frame_ball_hits = df_ball_pos[df_ball_pos['ball_hit'] == 1].index.tolist()
        return frame_ball_hits

    #Detect the ball
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
        results = self.model.predict(frame, conf = self.conf)[0]

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