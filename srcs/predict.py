import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

def predict_age(image_path, model_path, device):

    class_names = ['18-24', '25-39', '40-59', '60-plus']

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode!

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)


    with torch.no_grad():

        outputs = model(img_tensor) 
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    print("-" * 30)
    print(f"Image: {image_path}")
    print(f"Predicted Life Stage: {class_names[predicted_idx.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    print("-" * 30)

device = "cuda" if torch.cuda.is_available() else "cpu"

my_image = r"C:\Users\User\Documents\GitHub\DL_AgeEstimation\UTKFace_organized\18-24\18_0_0_20170110231228322.jpg.chip.jpg" 


my_model = "model_best_finetuned.pth" 

predict_age(my_image, my_model, device)