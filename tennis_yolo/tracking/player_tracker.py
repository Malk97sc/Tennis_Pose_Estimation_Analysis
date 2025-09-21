from ultralytics import YOLO
import cv2 as cv
import pickle
import numpy as np

class PlayerTracking:
    def __init__(self, model_path, max_dist = 70):
        """
        model_path: Path of YOLO 
        max_dist: Max distance in pixels to accept detections outside the HULL        
        """
        self.model = YOLO(model_path)
        self.max_dist = max_dist
    
    def _ref_point_status(self, box, hull):
        x1, y1, x2, y2 = map(int, box)
        feet_point = ((x1 + x2) // 2, y2)
        body_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        feet_inside = cv.pointPolygonTest(hull, feet_point, False)
        feet_dist   = cv.pointPolygonTest(hull, feet_point, True)
        body_inside = cv.pointPolygonTest(hull, body_point, False)
        body_dist   = cv.pointPolygonTest(hull, body_point, True)

        if feet_inside >= 0:
            return feet_point, "inside", 0.0
        elif body_inside >= 0:
            return body_point, "inside", 0.0
        else:
            if abs(feet_dist) <= abs(body_dist):
                ref_point = feet_point
                min_dist = abs(feet_dist)
            else:
                ref_point = body_point
                min_dist = abs(body_dist)

            if min_dist > self.max_dist:
                return None, "outside", min_dist
            return ref_point, f"dist {min_dist:.1f}", min_dist

    #Pick Players
    def choose_players(self, player_dt, hull, max_dist=None):
        if max_dist is None:
            max_dist = self.max_dist

        distances = []
        for tid, box in player_dt.items():
            ref_point, _, min_dist = self._ref_point_status(box, hull) #skip the status '_'
            if ref_point is not None:
                distances.append((tid, min_dist))

        if not distances:
            return []

        distances.sort(key=lambda x: x[1])

        players = [tid for tid, d in distances if d <= max_dist]
        if len(players) >= 2:
            return players[:2]
        fallback = [tid for tid, _ in distances][:2]
        return fallback

    def pick_players(self, hull, player_dt):
        if not player_dt:
            return []

        first_player_dt = player_dt[0]
        players = self.choose_players(first_player_dt, hull ,self.max_dist)
        players_filter = []

        for frame in player_dt:
            searched_players = {track_id: box for track_id, box in frame.items() if track_id in players}
            players_filter.append(searched_players)
        return players_filter

    #Player Detection
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
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_id = box.cls.tolist()[0]
            object_name = id_name[object_id]
            if object_name == "person":
                player[track_id] = result #only save the result boxes of the Person class by YOLO

        return player
    
    #Draw Boxes
    def draw_boxes(self, video_frames, player_detections, hull, color_box = (0, 255, 0)):
        output_frames = []

        for frame, players in zip(video_frames, player_detections):
            for track_id, box_pos in players.items():
                x1, y1, x2, y2 = map(int, box_pos)

                #Reference pts
                ref_point, status, _ = self._ref_point_status(box_pos, hull) #skip the min distance '_'s
                if ref_point is None:
                    continue

                #Draw
                cv.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                cv.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10),cv.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)
                #Reference point
                cv.circle(frame, ref_point, 6, (0, 255, 0), -1)
                cv.putText(frame, status, (ref_point[0], ref_point[1] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            output_frames.append(frame)

        return output_frames
