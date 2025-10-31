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

router = APIRouter()


@router.post("/detect/deepfake/video", response_class=FileResponse)
async def predict_video(file: UploadFile = File(...), frames: int = 50, background_tasks: BackgroundTasks = None):
    # Save uploaded video to temp dir
    # VideoSaver(file from user, temp directory path, unique video id generated)

    video_id = str(uuid.uuid4())
    try:
        video_path, video_folder = VideoSaver.save_file(file, ExtractionConfig.TEMP_DIR, video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {str(e)}")

    # Extract frames
    preprocessor = VideoPreprocessor(ExtractionConfig)
    stats = preprocessor.preprocess_frame(
        video_path=video_path,
        output_dir=video_folder,
        video_id=os.path.basename(video_folder),
        frames=frames
    )

    # If there are preprocessing errors, raise exception
    if stats.errors:
        raise HTTPException(status_code=400, detail={
            "video_id": stats.video_id,
            "message": "Video preprocessing failed",
            "errors": stats.errors
        })

    # Detect fake
    try:
        results = detect_deepfake(video_folder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deepfake detection failed: {str(e)}")

    # Process results
    def extract_float(value):
        """Recursively unwrap lists until a float/int is found."""
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return extract_float(value[0])
        return None  # skip invalid values

    scores = [extract_float(v) for v in results.values()]
    scores = [s for s in scores if s is not None]
    avg_score = float(sum(scores) / len(scores)) if scores else 0.0

    summary = {
        "video_id": video_id,
        "frames_analyzed": len(results),
        "average_score": round(avg_score, 4)
    }

    # Annotate images
    annotate_confidences(video_folder, results)

    # Create zip file of annotated results
    annotated_folder = Path(video_folder) / "annotated_results"
    
    if not annotated_folder.exists() or not any(annotated_folder.iterdir()):
        raise HTTPException(status_code=500, detail="Annotated results folder is empty or does not exist")
    
    # Create zip file in the video folder
    zip_filename = f"{video_id}_annotated_frames"
    zip_path = Path(video_folder) / zip_filename
    
    try:
        # shutil.make_archive returns the path with .zip extension
        zip_file_path = shutil.make_archive(
            base_name=str(zip_path),
            format='zip',
            root_dir=annotated_folder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create zip file: {str(e)}")

    # Schedule cleanup of the entire video folder after response is sent
    background_tasks.add_task(delayed_cleanup, video_folder, delay=60)

    # Return the zip file
    response = FileResponse(
        path=zip_file_path,
        media_type="application/zip",
        filename=f"{video_id}_annotated_frames.zip",
        background=background_tasks
    )

    # Add results summary to response headers
    response.headers["X-Frames-Analyzed"] = str(len(results))
    response.headers["X-Average-Score"] = str(round(avg_score, 4))

    # Optional: full JSON result as header (truncated if too large)
    # json_str = json.dumps(results)
    # if len(json_str) < 8000:  # headers limit ~8KB
    #     response.headers["X-Deepfake-Results"] = json_str
    # else:
    #     response.headers["X-Deepfake-Results"] = "Too large to include in headers"
    

    return response
