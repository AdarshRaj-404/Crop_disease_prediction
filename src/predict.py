import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import get_model
from dataset import get_dataloaders # just to get class names easily

def predict_image(image_path, model_path='best_model.pth'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Needs class names from somewhere. Easiest is from dataset or saved mapping.
    # For now, we'll quickly read dataloader to get classes (a bit slow but robust).
    # Ideally, class names should be saved in a JSON during training.
    try:
        _, _, _, classes = get_dataloaders()
    except Exception as e:
        print("Could not load dataset to get class names. Using dummy names.")
        classes = [f"Class_{i}" for i in range(23)]
        
    num_classes = len(classes)
    
    # Initialize Model and load weights
    model = get_model(num_classes=num_classes, pretrained=False)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.set_grad_enabled(False):
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, preds = torch.max(outputs, 1)
        
    predicted_class = classes[preds[0].item()]
    confidence = probabilities[preds[0].item()].item() * 100
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    return predicted_class, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict crop disease from an image.")
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model')
    args = parser.parse_args()
    
    model_path = os.path.join(os.path.dirname(__file__), '..', args.model_path)
    predict_image(args.image_path, model_path=model_path)
