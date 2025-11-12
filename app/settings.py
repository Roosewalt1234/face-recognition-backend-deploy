import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Supabase settings
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "service-images")
    MODEL_BUCKET: str = os.getenv("MODEL_BUCKET", "service-models")

    # Face recognition settings
    CONFIDENCE_PASS: float = float(os.getenv("CONFIDENCE_PASS", "80"))
    FACE_DATA_TABLE: str = os.getenv("FACE_DATA_TABLE", "face_recognition_data")
    EMPLOYEES_TABLE: str = os.getenv("EMPLOYEES_TABLE", "employees")
    ATTENDANCE_TABLE: str = os.getenv("ATTENDANCE_TABLE", "attendance_sheet")

    # Model settings
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")

    def __init__(self):
        if not self.SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not self.SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")

settings = Settings()