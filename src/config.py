import os, shutil

# Modified to step up one directory since this file now lives in src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST_FILE = "paper_list_72.txt"

def find_tesseract():
    env_path = os.getenv("TESSERACT_PATH") 
    if env_path and os.path.exists(env_path):
        return env_path
    if shutil.which("tesseract"): 
        return shutil.which("tesseract")
    win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
    ]
    for path in win_paths:
        if os.path.exists(path): return path
    linux_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract"
    ]
    for path in linux_paths:
        if os.path.exists(path): return path
    return None

CONFIG = {
    "EXAM_ID": "Exam_W25-ID_72",
    "INPUT_FILE": os.path.join(BASE_DIR, "data", "01_raw_pdfs", LIST_FILE),
    "TEMP_DIR": os.path.join(BASE_DIR, "temp_downloads"),
    "OUTPUT_IMG_DIR": os.path.join(BASE_DIR, "data", "02_extracted_dataset", "images_Exam_W25-ID_72"), 
    "LOG_DIR": os.path.join(BASE_DIR, "docs", "logs"),
    "TESSERACT_PATH": find_tesseract(),
    "TARGET_TOTAL_IMAGES": 250,
    "CLEANUP_INTERVAL": 50, 
    "CLIP_THRESHOLD": 0.16, 
    "CLIP_PROMPTS": [
        "a diagram of a quantum computing circuit with horizontal lines and gates",
        "a random photograph, selfie, animal, or natural scenery",
        "a blank white image",
        "a screenshot of code or text", 
        "a chart or graph that is not a quantum circuit"
    ],
    "DPI": 300,
    "CLUSTERING_THRESHOLD": 50,
    "SPLIT_X_DIST": 40,   
    "SPLIT_Y_DIST": 200,
    "CAPTION_BUFFER_PX": 3.5,
    "COLOR_BLOCK_THRESHOLD": 500,
    "CURVE_THRESHOLD": 150,
    "DENSITY_THRESHOLD": 0.5
}
