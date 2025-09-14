import torch 
import torchvision.transforms as transforms
import torchvision.models as models
import cv2 as cv

class CourtDetection:
    def __init__(self, model_path):
        self.model = models.resnet50(weights = None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location = 'cpu'))

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
