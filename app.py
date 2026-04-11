import os
import torch
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import io
from torchvision import transforms
from src.model import get_model
from src.dataset import get_dataloaders
from fastapi.middleware.cors import CORSMiddleware

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
DATA_DIR = os.path.join(os.path.dirname(__file__), 'New Plant Diseases Dataset')
if os.path.exists(DATA_DIR):
    classes = sorted(os.listdir(DATA_DIR))
else:
    # Fallback/Dummy classes if dataset folder is moved, just so API loads
    classes = [f"Class_{i}" for i in range(23)]

num_classes = len(classes)

# Load treatments mapping
try:
    with open(os.path.join(os.path.dirname(__file__), 'treatments.json'), 'r') as f:
        treatments_db = json.load(f)
except Exception as e:
    print("Warning: treatments.json not found:", e)
    treatments_db = {}


# Initialize Model globally so it stays in memory
try:
    model = get_model(num_classes=num_classes, pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded. Please ensure training is complete. Error: {e}")
    model = None

# Image transforms matcher
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def format_crop_name(c):
    name = c.split('___')[0].replace('_', ' ')
    if name == "Pepper, bell":
        name = "Pepper (Bell)"
    return name

@app.get("/crops")
async def get_crops():
    # Extract unique crop names from class names using helper
    crop_list = sorted(list(set([format_crop_name(c) for c in classes])))
    return JSONResponse(content={"crops": crop_list})

@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: str = Form(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized. Ensure training is done and best_model.pth exists.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, preds = torch.max(probabilities, 0)
            
        predicted_class = classes[preds.item()]
        predicted_crop = format_crop_name(predicted_class)
        
        # Validate that the predicted crop matches the user-selected crop
        if crop.lower() != predicted_crop.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Please upload a valid {crop} image. The model detected '{predicted_crop}' instead."
            )
        
        # Calculate human readable format for class names
        formatted_class = predicted_class.replace('___', ' - ').replace('_', ' ')
        
        # Get dynamic treatment plan
        treatment_steps = treatments_db.get(predicted_class, ["Consult a local agronomist for specific treatment."])
        
        return JSONResponse(content={
            "predicted_class": formatted_class,
            "raw_class": predicted_class,
            "confidence": float(confidence.item() * 100),
            "treatment": treatment_steps
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files to serve the frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
