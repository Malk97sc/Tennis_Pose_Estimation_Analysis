import torch 
import torchvision.transforms as transforms
import torchvision.models as models
import cv2 as cv
import numpy as np

class CourtDetection:
    def __init__(self, model_path):
        self.model = models.resnet50(weights = None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location = 'cpu', weights_only=True))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints
    
    def draw_keypoints_on_video(self, video_frames, keypoints, hull = None, show_court = False):
        output_video_frames = []
        kp = np.array(keypoints, dtype=int).reshape(-1, 2)
        
        court_edges = [
            (4, 5),   #top service line
            (6, 7),   #bottom service line
            (8, 9),   #net (top side)
            (10, 11), #net (bottom side)
            (12, 13)  #center service line
        ]

        for frame in video_frames:
            for i, (x, y) in enumerate(kp):
                cv.putText(frame, str(i), (x, y - 10),
                           cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                cv.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            if show_court:
                hull_int = hull.astype(int)
                cv.polylines(frame, [hull_int], isClosed=True, color=(0, 0, 255), thickness=2)
                
                for (i, j) in court_edges:
                    pt1, pt2 = tuple(kp[i]), tuple(kp[j])
                    cv.line(frame, pt1, pt2, (0, 0, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames