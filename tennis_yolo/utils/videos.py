import cv2 as cv

def read_video(source_path):
    cap = cv.VideoCapture(source_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_frames, video_path, fps = 25):
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(video_path, fourcc, fps, (output_frames[0].shape[1], output_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()
