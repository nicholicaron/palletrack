from pydantic import BaseModel


class OCRTextData(BaseModel):
    text: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class PalletData(BaseModel):
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    bbox: list[float]  # [x1, y1, x2, y2] - averaged or last known position
    ocr_texts: list[OCRTextData]


class VideoProcessResponse(BaseModel):
    success: bool
    message: str
    pallets: list[PalletData]
    total_frames: int
    fps: float
