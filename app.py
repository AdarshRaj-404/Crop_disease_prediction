import os
import torch
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import io
from torchvision import transforms
from src.model import get_model
from src.dataset import get_dataloaders
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel

app = FastAPI(title="Crop Disease Prediction API")

# Setup CORS just in case frontend is served elsewhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'best_model.pth'

# Load class names
# We'll fetch them from the directories. It's safer if you had them saved, 
# but for now we'll dynamically grab them.
DATA_DIR = r"c:\Users\WIN 10\Downloads\CROP DIS DATASET COMPRESSED (8)"
if os.path.exists(DATA_DIR):
    classes = sorted(os.listdir(DATA_DIR))
else:
    # Fallback/Dummy classes if dataset folder is moved, just so API loads
    classes = [f"Class_{i}" for i in range(23)]

num_classes = len(classes)

# Global placeholders for models
clip_model = None
clip_processor = None
clip_loading_attempted = False

def get_clip():
    global clip_model, clip_processor, clip_loading_attempted
    if not clip_loading_attempted and clip_model is None:
        clip_loading_attempted = True
        try:
            print("Attempting to load CLIP model (this may take a few minutes on first run)...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model loaded successfully.")
        except Exception as e:
            print(f"Warning: CLIP model could not be loaded. Error: {e}")
            clip_model = None
    return clip_model, clip_processor

# Initialize Model globally so it stays in memory
try:
    model = get_model(num_classes=num_classes, pretrained=False)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # Handle backward compatibility since we added Dropout layer
        if 'fc.weight' in state_dict:
            state_dict['fc.1.weight'] = state_dict.pop('fc.weight')
            state_dict['fc.1.bias'] = state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded. Please ensure training is complete. Error: {e}")
    model = None

# Image transforms matcher
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized. Ensure training is done and best_model.pth exists.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # --- REJECTION LAYER 1: Visual Heuristic Fallback (Local) ---
        # Code/Screenshots/Text have very high edge density compared to leaves
        from PIL import ImageFilter
        import numpy as np
        
        # 1. Edge Density Check
        grayscale = image.convert('L')
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        edge_data = np.array(edges)
        edge_density = np.mean(edge_data) / 255.0
        
        # 2. Color Balance Check (Leaves are usually dominantly green or have biological hues)
        # Code screens are often monochromatic (dark/white) or have high-contrast text.
        img_np = np.array(image)
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        # Calculate a "Greenness" factor
        green_mask = (g > r) & (g > b)
        green_ratio = np.sum(green_mask) / img_np.size * 3.0 # Multiplying by 3 because size is (H,W,3)
        
        # Heuristic: Code screens (especially dark mode) have extremely low green ratio 
        # and often very different edge densities.
        # Leaves usually have green_ratio > 0.1 even if diseased.
        # Screenshots of code usually have green_ratio < 0.05.
        
        is_likely_not_leaf = False
        if edge_density > 0.15: # Text/Code has very high edge density
            is_likely_not_leaf = True
        if green_ratio < 0.02 and edge_density > 0.08: # Low green and high edge density = likely code/text
            is_likely_not_leaf = True
            
        # --- REJECTION LAYER 2: CLIP (If available) ---
        is_clip_rejected = False
        curr_clip_model, curr_clip_processor = get_clip()
        if curr_clip_model is not None:
            clip_inputs = curr_clip_processor(
                text=["a photo of a plant leaf", "a photo of a random object, code screen, or scenery"],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
            with torch.set_grad_enabled(False):
                clip_outputs = curr_clip_model(**clip_inputs)
                clip_probs = clip_outputs.logits_per_image.softmax(dim=1)
                predicted_cat = clip_probs.argmax().item()
            
            if predicted_cat == 1:
                is_clip_rejected = True

        # Combine results: Reject if visually suspicious OR CLIP is certain
        if is_likely_not_leaf or is_clip_rejected:
            raise HTTPException(status_code=400, detail="please upload a valid leaf image")

        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, preds = torch.max(probabilities, 0)
            
        predicted_class = classes[preds.item()]
        
        # Calculate human readable format for class names
        formatted_class = predicted_class.replace('___', ' - ').replace('_', ' ')
        
        return JSONResponse(content={
            "predicted_class": formatted_class,
            "raw_class": predicted_class,
            "confidence": float(confidence.item() * 100)
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files to serve the frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
