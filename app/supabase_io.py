import io
from typing import List, Dict, Optional, Tuple
import requests
from supabase import create_client, Client

from .settings import settings

# Initialize Supabase client
sb: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

def list_face_images(employee_ids: Optional[List[str]] = None) -> List[Dict]:
    """List all face images from face_images table (multiple images per employee)"""
    try:
        query = sb.table("face_images").select("employee_id, image_path, image_index").eq("is_active", True)

        if employee_ids:
            query = query.in_("employee_id", employee_ids)

        result = query.execute()
        return result.data or []

    except Exception as e:
        print(f"Error listing face images: {e}")
        return []

def fetch_image_as_gray(image_path: str) -> Optional['np.ndarray']:
    """Fetch image from Supabase storage and convert to grayscale"""
    try:
        import cv2
        import numpy as np

        # Get public URL for the image
        public_url = sb.storage.from_(settings.SUPABASE_BUCKET).get_public_url(image_path)

        # Download image
        response = requests.get(public_url, timeout=10)
        response.raise_for_status()

        # Convert to numpy array
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Failed to decode image: {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray

    except Exception as e:
        print(f"Error fetching image {image_path}: {e}")
        return None

def upload_model_files(yml_bytes: bytes, label_npy: bytes, label_json: str):
    """Upload trained model files to Supabase storage"""
    try:
        # Upload YML model file
        yml_path = f"{settings.MODEL_DIR}/lbph_model.yml"
        sb.storage.from_(settings.MODEL_BUCKET).upload(
            yml_path,
            yml_bytes,
            {"content-type": "application/octet-stream", "upsert": "true"}
        )

        # Upload label NPY file
        npy_path = f"{settings.MODEL_DIR}/labels.npy"
        sb.storage.from_(settings.MODEL_BUCKET).upload(
            npy_path,
            label_npy,
            {"content-type": "application/octet-stream", "upsert": "true"}
        )

        # Upload label JSON file
        json_path = f"{settings.MODEL_DIR}/labels.json"
        sb.storage.from_(settings.MODEL_BUCKET).upload(
            json_path,
            label_json.encode('utf-8'),
            {"content-type": "application/json", "upsert": "true"}
        )

        print("Model files uploaded successfully")

    except Exception as e:
        print(f"Error uploading model files: {e}")
        raise

def download_model_files() -> Tuple[Optional[bytes], Optional[bytes]]:
    """Download model files from Supabase storage"""
    try:
        # Download YML model file
        yml_path = f"{settings.MODEL_DIR}/lbph_model.yml"
        yml_response = sb.storage.from_(settings.MODEL_BUCKET).download(yml_path)
        yml_bytes = yml_response if yml_response else None

        # Download label NPY file
        npy_path = f"{settings.MODEL_DIR}/labels.npy"
        npy_response = sb.storage.from_(settings.MODEL_BUCKET).download(npy_path)
        npy_bytes = npy_response if npy_response else None

        if yml_bytes and npy_bytes:
            print("Model files downloaded successfully")
            return yml_bytes, npy_bytes
        else:
            print("Model files not found in storage")
            return None, None

    except Exception as e:
        print(f"Error downloading model files: {e}")
        return None, None

def get_employee_map(employee_ids: List[str]) -> Dict[str, str]:
    """Get employee name mapping for given employee IDs"""
    try:
        result = sb.table(settings.EMPLOYEES_TABLE).select("id, full_name").in_("id", employee_ids).execute()

        employee_map = {}
        for emp in result.data or []:
            employee_map[emp["id"]] = emp["full_name"] or f"Employee {emp['id'][:8]}"

        return employee_map

    except Exception as e:
        print(f"Error getting employee map: {e}")
        return {}