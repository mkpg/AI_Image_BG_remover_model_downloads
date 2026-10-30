import io
import os
import torch
from PIL import Image

# Force the AI model to be downloaded exactly into an "ai_models" folder right next to this script!
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['HF_HOME'] = os.path.join(current_dir, "ai_models")

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
from transformers import pipeline

app = FastAPI()

# Enable CORS so your React app can communicate with this local API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*70)
print("Loading Enterprise AI Model (briaai/RMBG-1.4)...")
print("NOTE: The first time you run this, it will download the model files (~170MB) to your PC.")
print("After that, it will load instantly from your local hard drive forever!")
print("="*70 + "\n")

# Use GPU if available on your PC, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)

print("\n✅ Model loaded successfully into your local PC!\n")

@app.post("/remove_bg")
async def remove_bg(file: UploadFile = File(...)):
    print(f"\n---> Processing image: {file.filename}")
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Run the model locally
    output_image = pipe(input_image)
    
    # Save the result to memory
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    print("---> Done processing! Returning perfectly cut out image.\n")
    return Response(content=img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    print("\n" + "🟢 "*20)
    print("LOCAL AI BACKGROUND REMOVAL SERVER IS RUNNING!")
    print("API URL: http://localhost:8000")
    print("Paste this URL into the 'Enterprise AI Backend' box in your React App!")
    print("🟢 "*20 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
