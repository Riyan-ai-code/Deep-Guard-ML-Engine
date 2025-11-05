import os
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.services.model import detect_deepfake
from app.config.config import ExtractionConfig
from fastapi.responses import FileResponse
from app.services.save_video import VideoSaver
from app.services.video_preprocessor import VideoPreprocessor
from fastapi import HTTPException
from app.utils.delayed_cleanup import delayed_cleanup
import json
from app.utils.annotate_images import annotate_confidences
from fastapi.responses import JSONResponse

router = APIRouter()



@router.post("/detect/deepfake/video", response_class=FileResponse)
async def predict_video(file: UploadFile = File(...), frames: int = 50, background_tasks: BackgroundTasks = None):
    video_id = str(uuid.uuid4())
    video_folder = None
    
    try:
        # Save uploaded video
        video_path, video_folder = VideoSaver.save_file(file, ExtractionConfig.TEMP_DIR, video_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to save uploaded video: {str(e)}"}
        )

    # Extract frames
    preprocessor = VideoPreprocessor(ExtractionConfig)
    stats = preprocessor.preprocess_frame(
        video_path=video_path,
        output_dir=video_folder,
        video_id=os.path.basename(video_folder),
        frames=frames
    )

    # If preprocessing errors, schedule cleanup and return error
    if stats.errors:
        background_tasks.add_task(delayed_cleanup, video_folder, delay=60)
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "video_id": stats.video_id,
                    "message": "Video preprocessing failed",
                    "errors": stats.errors
                }
            }
        )

    # Detect fake
    try:
        results = detect_deepfake(video_folder)
    except Exception as e:
        background_tasks.add_task(delayed_cleanup, video_folder, delay=60)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Deepfake detection failed: {str(e)}"}
        )

    # Process results
    def extract_float(value):
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return extract_float(value[0])
        return None

    scores = [extract_float(v) for v in results.values()]
    scores = [s for s in scores if s is not None]
    avg_score = float(sum(scores) / len(scores)) if scores else 0.0

    # Annotate images
    annotate_confidences(video_folder, results)

    # Create zip file
    annotated_folder = Path(video_folder) / "annotated_results"
    
    if not annotated_folder.exists() or not any(annotated_folder.iterdir()):
        background_tasks.add_task(delayed_cleanup, video_folder, delay=60)
        return JSONResponse(
            status_code=500,
            content={"detail": "Annotated results folder is empty or does not exist"}
        )
    
    zip_filename = f"{video_id}_annotated_frames"
    zip_path = Path(video_folder) / zip_filename
    
    try:
        zip_file_path = shutil.make_archive(
            base_name=str(zip_path),
            format='zip',
            root_dir=annotated_folder
        )
    except Exception as e:
        background_tasks.add_task(delayed_cleanup, video_folder, delay=60)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to create zip file: {str(e)}"}
        )

    # Schedule cleanup for success case
    background_tasks.add_task(delayed_cleanup, video_folder, delay=60)

    # Return the zip file
    response = FileResponse(
        path=zip_file_path,
        media_type="application/zip",
        filename=f"{video_id}_annotated_frames.zip",
        background=background_tasks
    )

    response.headers["X-Frames-Analyzed"] = str(len(results))
    response.headers["X-Average-Score"] = str(round(avg_score, 4))

    return response
