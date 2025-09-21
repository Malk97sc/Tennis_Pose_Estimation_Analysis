import cv2 as cv

from pathlib import Path

from utils import read_video, save_video
from utils import DATA_DIR, RAW_DATA_DIR, MODELS_DIR

from tracking import PlayerTracking, BallTracking
from court import CourtDetection
from utils import refine_keypoints_subpix, compute_player_stats, draw_player_stats

def main():
    video_path = RAW_DATA_DIR / "input_video.mp4" 
    output_path = Path(DATA_DIR) / "results" / "video"
    output_path.mkdir(parents=True, exist_ok=True)
    
    yolo_model = "yolo11x.pt"
    player_model_path = MODELS_DIR / "yolo" / yolo_model
    ball_model_path = MODELS_DIR / "fine_tuning" / "ball_yolo11x" / "weights" / "best.pt"
    court_lines_model_path = MODELS_DIR / "court_detection" / "keypoints_model.pth"
    stub_path = MODELS_DIR / "tracker_stubs"    

    #--------read video-------
    out_fps = 25
    height, width, fps, video = read_video(video_path)
    print(f"Height: {height}, Width: {width}")
    print(f"FPS: {fps}")

    #--------players----------
    player_track = PlayerTracking(player_model_path) #player tracking instance
    player_dt = player_track.detect_player(video, read_stub = True, stub_path = stub_path / "player_detection_y11.pkl") #player detection
    
    #--------ball-------------
    ball_track = BallTracking(ball_model_path, fps, confidence = 0.25) #ball tracking instance
    ball_dt = ball_track.detect_ball(video, read_from_stub = True, stub_path = stub_path / "ball_detection_y11.pkl") #ball detection
    interpolation_method = 'linear'
    order = 3 #this orders is only for Spline, cubicspline and Polynomial interpolation
    ball_dt = ball_track.interpolate_ball(ball_dt, method = interpolation_method, order = order) #ball interpolation to improve the result

    #--------ball hit---------
    ball_hit = ball_track.get_shot_frames(ball_dt, changes = 25)
    #print(ball_hit)

    #-------court lines------------
    first_frame = video[0]
    court_line = CourtDetection(court_lines_model_path)
    court_kp = court_line.predict(first_frame)
    #print(f"raw: {court_kp}")
    
    #-------PostProcessing--------
    court_kp, _ = refine_keypoints_subpix(first_frame, keypoints = court_kp, win_size = 40)
    #print(f"refined: {court_kp}")

    #------pick Players---------
    player_dt = player_track.pick_players(court_kp, player_dt)

    #------stats---------------
    player_stats_df = compute_player_stats(player_dt, ball_dt, ball_hit, court_kp, out_fps)

    #-------draw boxes----------
    out_video = player_track.draw_boxes(video, player_dt, court_kp) 
    out_video = ball_track.draw_boxes(out_video, ball_dt)
    #out_video = court_line.draw_keypoints_on_video(out_video, court_kp)

    #------draw stats----------
    out_video = draw_player_stats(out_video, player_stats_df)

    output_video_path = output_path / f"output_video_{yolo_model}_{interpolation_method}.avi"
    save_video(out_video, output_video_path, out_fps)
    print(f"Save video in: {output_video_path}")

if __name__ == "__main__":
    main()