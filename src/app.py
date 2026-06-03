import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.inference import InferencePipeline
from src.config import Config

app = FastAPI(
    title="The Silent Cry Decoder API", 
    description="Backend API for classifying baby cry reasons using a 3-stream feature fusion neural network.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance initialized on startup
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    # Initialize the inference pipeline, loading the best checkpoint if available
    pipeline = InferencePipeline()

@app.get("/health")
def health_check():
    """Health check endpoint."""
    checkpoint_exists = os.path.exists(os.path.join(Config.SAVE_DIR, "best_model.pth"))
    return {
        "status": "healthy",
        "device": Config.DEVICE,
        "model_loaded": pipeline is not None,
        "checkpoint_available": checkpoint_exists
    }

@app.post("/predict")
async def predict_cry(file: UploadFile = File(...)):
    """
    Endpoint to predict the reason of baby crying from an audio file.
    Supports audio uploads. Converts, extracts features, and processes.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading or uninitialized.")
        
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    # Supported formats (librosa handles wav, mp3, ogg, etc. if soundfile/audioread are present)
    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".3gp", ".caf", ".aac"}
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Must be one of {allowed_extensions}")
        
    temp_dir = os.path.join(Config.BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    
    try:
        # Save upload to temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run the inference pipeline
        result = pipeline.predict(temp_file_path)
        
        # Add metadata for response
        result["filename"] = filename
        return result
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    finally:
        # Guarantee temporary file is deleted to avoid disk clutter
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error removing temp file {temp_file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    # Allow running server directly
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
