from pathlib import Path

from utils import read_video, save_video
from utils import DATA_DIR, RAW_DATA_DIR, MODELS_DIR
from tracking import PlayerTracking, BallTracking

def main():
    video_path = RAW_DATA_DIR / "test_video.mp4"
    output_path = Path(DATA_DIR) / "results" / "video"
    output_path.mkdir(parents=True, exist_ok=True)
    
    player_model_path = MODELS_DIR / "yolo" / "yolov8x.pt"
    ball_model_path = MODELS_DIR / "fine_tuning" / "ball_yolov8x" / "weights" / "best.pt"
    stub_path = MODELS_DIR / "tracker_stubs"    

    #read video
    _, _, fps, video = read_video(video_path)
    print(f"FPS: {fps}")

    #players
    player_track = PlayerTracking(player_model_path) #player tracking instance
    player_dt = player_track.detect_player(video, read_stub = True, stub_path = stub_path / "player_detection.pkl") #player detection
    
    #ball
    ball_track = BallTracking(ball_model_path, fps) #ball tracking instance
    ball_dt = ball_track.detect_ball(video, read_from_stub = True, stub_path = stub_path / "ball_detection.pkl") #ball detection
    interpolation_method = 'spline'
    order = 3 #this orders is only for Spline and Polynomial interpolation
    ball_dt = ball_track.interpolate_ball(ball_dt, method = interpolation_method, order) #ball interpolation to improve the result

    #draw boxes
    out_video = player_track.draw_boxes(video, player_dt) 
    out_video = ball_track.draw_boxes(video, ball_dt)

    output_video_path = output_path / f"output_video_{interpolation_method}.avi"
    save_video(out_video, output_video_path)
    print(f"Save video in: {output_video_path}")

if __name__ == "__main__":
    main()