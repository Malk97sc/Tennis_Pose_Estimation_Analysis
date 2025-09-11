from pathlib import Path

from utils import read_video, save_video
from utils import DATA_DIR, RAW_DATA_DIR, MODELS_DIR
from tracking import PlayerTracking

def main():
    video_path = RAW_DATA_DIR / "input_video.mp4"
    output_path = Path(DATA_DIR) / "results" / "video"
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / "yolo" / "yolov8x.pt"
    stub_path = MODELS_DIR / "tracker_stubs"    

    #read video
    video = read_video(video_path)    

    #players
    player_track = PlayerTracking(model_path) #player tracking instance
    player_dt = player_track.detect_player(video, 
                                           read_stub = True,
                                           stub_path = stub_path / "player_detection.pkl") #player detection
    video = player_track.draw_boxes(video, player_dt) 

    output_video_path = output_path / "output_video.avi"
    save_video(video, output_video_path)

if __name__ == "__main__":
    main()