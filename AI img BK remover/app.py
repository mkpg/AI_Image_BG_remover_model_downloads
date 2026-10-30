import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import gradio as gr
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Enterprise AI Model (briaai/RMBG-1.4)...")
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device="cpu")

@app.post("/remove_bg")
async def remove_bg(file: UploadFile = File(...)):
    print("Processing image...")
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    output_image = pipe(input_image)
    
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

demo = gr.Interface(
    fn=lambda x: x, 
    inputs="image", 
    outputs="image", 
    title="🟢 Skytree AI API is Online!",
    description="Your private FastAPI server is running perfectly."
)

app = gr.mount_gradio_app(app, demo, path="/")
