from ultralytics import YOLO
import cv2 as cv
import pickle

class PlayerTracking:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def _get_center(self, box):
        x1, y1, x2, y2 = box
        center_x = (x1+x2) // 2
        center_y = (y1+y2) // 2
        return center_x, center_y
    
    def _distance(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]+p2[1])**2)**0.5

    def pick_players(self, court_kp, player_dt):
        first_player_dt = player_dt[0]
        players = self.choose(court_kp, first_player_dt)
        players_filter = []
        
        for player_dict in player_dt: #we gonna search the players in the choose players
            searched_players_filter = {track_id: box for track_id, box in player_dict.items() if track_id in players}
            players_filter.append(searched_players_filter)
        return players_filter

    
    def choose(self, court_kp, player_dt):
        distances = []
        for id, box in player_dt.items():
            cx, cy = self._get_center(box)

            min_dist = -999.9
            for i in range(0, len(court_kp), 2):
                court_points = (court_kp[i], court_kp[i+1])
                distance = self._distance((cx, cy), court_points)
                if distance < min_dist:
                    min_dist = distance 
            distances.append((id, min_dist))

        distances.sort(key = lambda x: x[1])
        players = [distances[0][0], distances[1][0]]
        return players

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
        for frame, player in zip(video_frames, player_detections):
            for track_id, box_pos in player.items():
                x1, y1, x2, y2 = box_pos
                cv.putText(frame, f"Player ID: {track_id}", (int(box_pos[0]), int(box_pos[1] -10 )), cv.FONT_HERSHEY_COMPLEX, 0.9, color_box, 2)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
            output_frames.append(frame)
        
        return output_frames