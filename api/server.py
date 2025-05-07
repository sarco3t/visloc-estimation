from io import BytesIO
import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from inference import load_model, run_evaluation
app = FastAPI()

# Load the model and background collection once at startup
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
background_path = os.path.join(base_dir, "../models/back_coll_features.hdf5")
model, transform, mus, cells_assignments, back_col_emb, back_col_cells = load_model(
    use_cpu=False, gpu=0, background_path=background_path
)

@app.post("/evaluate/")
async def evaluate_http(image: UploadFile):
    try:
        # Read the uploaded image
        image_data = await image.read()
        image_file = BytesIO(image_data)

        # Validate the image
        try:
            Image.open(image_file).verify()  # Verify if the file is a valid image
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Prepare arguments
        class Args:
            use_cpu = False
            gpu = 0
            top_k = 10
            eps = 1.0
            conf_scale = 25

        # Run evaluation
        result = run_evaluation(
            model, transform, mus, cells_assignments, back_col_emb, back_col_cells, Args, image=image_file
        )
        return JSONResponse(content={
            "prediction": {
                "latitude": result["prediction"]["latitude"],
                "longitude": result["prediction"]["longitude"]
            },
            "confidence": float(result["confidence"])
        })
    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)