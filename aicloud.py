from fastapi import FastAPI, Header, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware # ★ ADDED: Import for CORS
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import logging
import time
from datetime import datetime
import os  # Necessary for os.getenv

# --- Configuration (Simulated Pydantic Settings) ---
class Settings:
    # Use environment variable if available, otherwise default to "1234"
    API_KEY: str = os.getenv("SUZI_API_KEY", "1234")
    DATA_DIR_NAME: str = "data"
    LIVE_DATA_CACHE_TTL: int = 5  # Cache live_data GET requests for 5 seconds

# Initialize settings and data directory
settings = Settings()
DATA_DIR: Path = Path(__file__).parent / settings.DATA_DIR_NAME
DATA_DIR.mkdir(exist_ok=True)

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='suzi_cloud.log', filemode='a')
logger = logging.getLogger(__name__)

# --- Caching State ---
live_data_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Suzi Cloud",
    description="A simple, secure, file-based JSON store with a dedicated endpoint for YOLO-style live data.",
    version="1.2.1",
    on_startup=[lambda: logger.info("Suzi Cloud API starting up...")],
    on_shutdown=[lambda: logger.info("Suzi Cloud API shutting down.")]
)

# ★ CORS Configuration to handle client-side preflight (OPTIONS) requests
origins = [
    "*", # WARNING: Allows ALL origins. Use specific client URLs in production for security.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, OPTIONS)
    allow_headers=["*", "X-Api-Key"], # Crucial for allowing the custom X-Api-Key header
)
# --- END CORS CONFIGURATION ---

# --- Dependency: API Key Authentication ---
def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Checks for a valid API key in the 'X-Api-Key' header."""
    if x_api_key is None or x_api_key != settings.API_KEY:
        logger.warning(f"Unauthorized access attempt with key: {x_api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Please provide a valid 'X-Api-Key' header."
        )
    return x_api_key

# --- Storage Service ---
class FileStorageService:
    """A service class to encapsulate file and memory operations."""
    def __init__(self, data_dir: Path, json_storage: Dict[str, Any]):
        self.data_dir = data_dir
        self.storage = json_storage

    def get_path(self, name: str) -> Path:
        """Generates a safe path for a JSON file."""
        safe_name = "".join(c for c in name if c.isalnum() or c == "_").lower()
        if not safe_name:
            raise ValueError("Invalid file name provided.")
        return self.data_dir / f"{safe_name}.json"

    def save(self, name: str, data: Any):
        """Saves data to a JSON file and updates memory storage."""
        try:
            path = self.get_path(name)
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.storage[name] = data
            logger.info(f"Saved data for '{name}' to disk and memory.")
        except (IOError, ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to save '{name}' data: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to save data: {e}"
            )

    def load_from_disk(self, name: str) -> Optional[Any]:
        """Loads data from a JSON file, handles errors."""
        try:
            path = self.get_path(name)
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load/decode JSON for '{name}': {e}")
            return None

    def get(self, name: str) -> Optional[Any]:
        """Retrieves data from memory, attempting to load from disk if missing."""
        if name in self.storage:
            return self.storage[name]

        loaded = self.load_from_disk(name)
        if loaded is not None:
            self.storage[name] = loaded
            return loaded
        return None

    def delete(self, name: str):
        """Deletes data from memory and disk."""
        if name in self.storage:
            self.storage.pop(name)

        try:
            path = self.get_path(name)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file for '{name}'.")
        except (IOError, ValueError) as e:
            logger.error(f"Failed to delete file for '{name}': {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to delete file: {e}"
            )

# --- Global Storage and Initialization ---
_json_storage: Dict[str, Any] = {}

def load_all_json_files_initial():
    """Initial loading logic run once on startup."""
    initial_storage: Dict[str, Any] = {}
    temp_service = FileStorageService(DATA_DIR, initial_storage)

    for file in DATA_DIR.glob("*.json"):
        name = file.stem
        loaded_data = temp_service.load_from_disk(name)
        if loaded_data is not None:
            initial_storage[name] = loaded_data

    logger.info(f"Loaded {len(initial_storage)} JSON files from disk on startup.")
    return initial_storage

_json_storage = load_all_json_files_initial()

def get_storage_service() -> FileStorageService:
    """FastAPI Dependency that provides the singleton Storage Service."""
    return FileStorageService(DATA_DIR, _json_storage)

# --- Core Models ---
# Hides this model from the top-level schema list
class BBox(BaseModel):
    x1: int = Field(..., ge=0, description="Top-left x-coordinate.")
    y1: int = Field(..., ge=0, description="Top-left y-coordinate.")
    x2: int = Field(..., ge=0, description="Bottom-right x-coordinate.")
    y2: int = Field(..., ge=0, description="Bottom-right y-coordinate.")
    
    model_config = {
        "json_schema_extra": {
            "hidden": True
        }
    }

# Hides this model from the top-level schema list
class LiveObject(BaseModel):
    label: str = Field(..., description="Detected object class label (e.g., 'person', 'car').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0.")
    bbox: BBox
    
    model_config = {
        "json_schema_extra": {
            "hidden": True
        }
    }

# ★ 1フレーム分の data 
class LiveDataEntry(BaseModel):
    timestamp_unix: float = Field(..., description="Unix timestamp of the detection.")
    timestamp_readable: str = Field(..., description="Human-readable timestamp string (e.g., ISO format).")
    source: Optional[str] = Field(None, description="Optional identifier for the camera/source.")
    objects: List[LiveObject]

# ★ history 全体 
# ★ history 全体 + location
class LiveDataHistory(BaseModel):
    location: Optional[str] = Field(
        None,
        description="Location name where this live data is captured (e.g. 'Main Tower 3F')."
    )
    history: List[LiveDataEntry] = Field(
        ..., description="List of recent YOLO detection frames (e.g., last 10 frames)."
    )

# Initialize current_live_data with optional validation
current_live_data: Optional[LiveDataHistory] = None

if "live_data" in _json_storage:
    try:
        current_live_data = LiveDataHistory.model_validate(_json_storage["live_data"])
        live_data_cache["data"] = current_live_data.model_dump(exclude_none=True)
        live_data_cache["timestamp"] = time.time()
        logger.info("Successfully loaded and validated initial 'live_data' as history.")
    except ValidationError as e:
        logger.warning(f"Failed to validate 'live_data' on startup, ignoring: {e}")
        current_live_data = None
        del _json_storage["live_data"]
    except Exception as e:
        logger.error(f"Unexpected error during live_data hydration: {e}")
        current_live_data = None
        del _json_storage["live_data"]

# ------------------------------------
#       ENDPOINTS
# ------------------------------------

# Monitoring
@app.get("/health", tags=["Monitoring"], status_code=status.HTTP_200_OK)
def health_check():
    """Simple health check for load balancers and monitoring systems."""
    current_time = datetime.now().isoformat()
    return {
        "status": "ok",
        "api_version": app.version,
        "time": current_time
    }

# --- YOLO live_data Endpoints (for streaming updates) ---
@app.put("/live_data", tags=["YOLO Live Data"])
def put_live_data(
    live_data: LiveDataHistory,
    api_key: str = Depends(get_api_key),
    storage: FileStorageService = Depends(get_storage_service)
):
    """
    Update the current YOLO live data feed (history of frames) by sending a JSON body.
    """
    global current_live_data, live_data_cache
    current_live_data = live_data

    data_dict = live_data.model_dump(mode="json", exclude_none=True)

    # Save to disk as 'live_data.json'
    storage.save("live_data", data_dict)

    # Update the cache immediately on write
    live_data_cache["data"] = data_dict
    live_data_cache["timestamp"] = time.time()

    return {"ok": True, "message": "Live data history updated successfully (JSON body)."}

# ★ GENERIC JSON FILE UPLOAD ENDPOINT (RENAMED)
@app.post("/data_file", tags=["Generic Store - File Upload"])
async def post_json_file_upload(
    # The field name 'file' must match the client's FormData.append('file', ...)
    file: UploadFile = File(..., description="A JSON file to be stored in the generic store. It will be saved under its filename (without extension)."),
    api_key: str = Depends(get_api_key), # Authentication is required
    storage: FileStorageService = Depends(get_storage_service)
):
    """
    Receive a JSON file via multipart/form-data and store its content 
    under the name derived from the filename (e.g., 'data.json' -> store name 'data').
    This endpoint is separate from the time-series 'live_data' stream.
    """
    if not file.filename or not file.filename.lower().endswith('.json'):
        # Note: file.filename can be None in some contexts, so check first
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File validation failed: Only .json files are accepted. Received: {file.filename}"
        )
    
    # New logic: Determine the store name from the filename stem
    store_name = Path(file.filename).stem 
    if not store_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not derive a valid store name from the filename.")

    try:
        # 1. Read the file content
        content = await file.read()
        json_data = json.loads(content)
        
        # 2. Store the data under the file's stem name
        storage.save(store_name, json_data)
        
        return {
            "ok": True, 
            "message": f"File uploaded and stored successfully as '{store_name}' in the generic JSON store.",
            "filename": file.filename,
            "stored_name": store_name,
            "size": len(content)
        }

    except json.JSONDecodeError:
        logger.error(f"Failed to decode uploaded file '{file.filename}' as JSON.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON format in file: {file.filename}. Please check the file content."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during file processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the file: {e}"
        )
    finally:
        await file.close()

@app.get("/live_data", tags=["YOLO Live Data"], response_model=Dict[str, Any])
def get_live_data():
    """
    Retrieve the current YOLO live data history, using a short-term cache.
    """
    global live_data_cache

    # Check Cache TTL
    time_since_last_cached = time.time() - live_data_cache["timestamp"]
    if live_data_cache["data"] is not None and time_since_last_cached < settings.LIVE_DATA_CACHE_TTL:
        logger.info(f"Cache hit for /live_data (stale by {time_since_last_cached:.2f}s)")
        return {"ok": True, "data": live_data_cache["data"], "cached": True}

    if current_live_data is None:
        return {"ok": True, "data": None, "message": "No live data currently stored."}

    # Cache miss - re-read from memory and update cache
    data_to_send = current_live_data.model_dump(exclude_none=True)
    live_data_cache["data"] = data_to_send
    live_data_cache["timestamp"] = time.time()

    logger.info("Cache miss for /live_data - updating cache.")
    return {"ok": True, "data": data_to_send, "cached": False}

# --- Generic Multi-File JSON API ---
@app.put("/json/{name}", tags=["Generic Store"], status_code=status.HTTP_200_OK)
def put_json(
    name: str,
    data: Dict[str, Any],
    api_key: str = Depends(get_api_key),
    storage: FileStorageService = Depends(get_storage_service)
):
    """
    Store arbitrary JSON data under a unique name (key) via JSON body.
    """
    try:
        storage.get_path(name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    storage.save(name, data)
    return {"ok": True, "name": name, "message": f"'{name}' data stored successfully."}

@app.get("/json/{name}", tags=["Generic Store"], response_model=Dict[str, Any])
def get_json(name: str, storage: FileStorageService = Depends(get_storage_service)):
    """
    Retrieve JSON data stored under the given name.
    """
    try:
        storage.get_path(name)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid name format.")

    data = storage.get(name)

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data store '{name}' not found."
        )

    return {"ok": True, "name": name, "data": data}

@app.get("/json", tags=["Generic Store"], response_model=Dict[str, Any])
def list_json(storage: FileStorageService = Depends(get_storage_service)):
    """
    List all currently stored JSON document names (keys).
    """
    return {"ok": True, "names": sorted(list(storage.storage.keys()))}

@app.delete("/json/{name}", tags=["Generic Store"], status_code=status.HTTP_200_OK)
def delete_json(
    name: str,
    api_key: str = Depends(get_api_key),
    storage: FileStorageService = Depends(get_storage_service)
):
    """
    Delete JSON data stored under the given name from memory and disk.
    """
    try:
        storage.get_path(name)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid name format.")

    if storage.get(name) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data store '{name}' not found."
        )

    storage.delete(name)

    return {"ok": True, "name": name, "message": f"'{name}' data deleted successfully."}

# --- ENDPOINT TO GET ALL DATA ---
@app.get("/all_json", tags=["Generic Store"])
def get_all_json(storage: FileStorageService = Depends(get_storage_service)):
    """
    Return ALL JSON documents as one big dict (the entire in-memory store).
    """
    return storage.storage

# --- Run with: python suzi_cloud.py ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
