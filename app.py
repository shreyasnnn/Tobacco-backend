#uvicorn app:app --reload
import os
import uuid
from datetime import datetime
from mangum import Mangum
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
import tempfile

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# CONFIG
# ======================
SUPABASE_URL = "https://ildeffyvkiaytvktamga.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlsZGVmZnl2a2lheXR2a3RhbWdhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAwNzQ3OCwiZXhwIjoyMDcwNTgzNDc4fQ.NtKHxwReJnGM4p_rE8nLsRgW27snfmngiF5MwCTjUOo"
BUCKET_NAME = "tobacco_uploads"

# Create Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Supabase client created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create Supabase client: {e}")
    raise

# Load model
try:
    model = load_model("mobilenetv2_tobacco_model.h5")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# PREDICTION ONLY
# ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        logger.info(f"📤 Running prediction for: {file.filename}")

        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read file
        file_bytes = await file.read()

        # Save temporarily for model
        filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Preprocess for model
        img = image.load_img(temp_path, target_size=(640, 640))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        confidence = float(np.max(preds) * 100)
        class_index = int(np.argmax(preds))
        classes = ['ChengShu', 'JiaShu', 'QianShu', 'ShangShu']
        result = classes[class_index]

        logger.info(f"🎯 Prediction: {result} ({confidence:.2f}%)")

        return {
            "result": result,
            "confidence": confidence,
            "message": "Prediction completed successfully"
        }

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {"error": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"🧹 Cleaned up temp file: {temp_path}")

# ======================
# SAVE TO HISTORY
# ======================
@app.post("/save")
async def save_to_history(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    result: str = Form(...),
    confidence: float = Form(...)
):
    try:
        logger.info(f"💾 Saving to history for user {user_id}")

        # Read file into memory
        file_bytes = await file.read()

        # Create structured folder path: user/{uid}/image/{timestamp_filename}
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        safe_filename = f"{timestamp}_{uuid.uuid4()}_{file.filename}"
        folder_path = f"user/{user_id}/images/{safe_filename}"

        # Upload to Supabase storage
        upload_response = supabase.storage.from_(BUCKET_NAME).upload(
            path=folder_path,
            file=file_bytes,
            file_options={"content-type": file.content_type}
        )

        # Check if upload failed
        if hasattr(upload_response, "error") and upload_response.error:
            raise Exception(f"Storage upload failed: {upload_response.error}")

        # Build public URL
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{folder_path}"

        # Insert record into DB
        insert_data = {
            "user_id": user_id,
            "image_url": image_url,
            "result": result,
            "confidence": confidence,
            "status": "processed",
            "processed_at": datetime.utcnow().isoformat()
        }
        insert_response = supabase.table("upload_history").insert(insert_data).execute()

        if hasattr(insert_response, "error") and insert_response.error:
            raise Exception(f"Database insert failed: {insert_response.error}")

        logger.info(f"✅ File saved successfully at {folder_path}")
        return {
            "message": "Saved to history successfully",
            "image_url": image_url,
            "path": folder_path
        }

    except Exception as e:
        logger.error(f"❌ Save to history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================
# DEBUGGING ENDPOINTS
# ======================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/test-db")
async def test_database():
    try:
        response = supabase.table("upload_history").select("*").limit(1).execute()
        return {"status": "database connected", "data": response.data}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test-storage")
async def test_storage():
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
        return {"status": "storage connected", "files": files}
    except Exception as e:
        return {"error": str(e)}


# Wrap FastAPI for serverless
handler = Mangum(app)