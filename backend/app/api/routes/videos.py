import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import VideoProcessResponse
from app.services import get_video_processor

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/process", response_model=VideoProcessResponse)
async def process_video(
    file: Annotated[UploadFile, File(description="Video file to process")]
) -> VideoProcessResponse:
    """
    Upload and process a video to detect and track pallets.

    Returns:
    - Detected pallets with tracking IDs
    - OCR text extracted from each pallet
    - Frame metadata (total frames, FPS)
    """
    # Validate file type by extension and/or content type
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    file_ext = Path(file.filename or "").suffix.lower() if file.filename else ""

    has_valid_content_type = file.content_type and file.content_type.startswith("video/")
    has_valid_extension = file_ext in valid_extensions

    if not (has_valid_content_type or has_valid_extension):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Please upload a video file. Got content_type: {file.content_type}, extension: {file_ext}"
        )

    # Save uploaded file to temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            # Read and write file content
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        # Process video
        processor = get_video_processor()
        result = processor.process_video(tmp_path)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        ) from e

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()
