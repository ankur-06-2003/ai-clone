from fastapi import FastAPI, UploadFile, File, Depends, HTTPException,Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
# Synchronous MongoDB driver
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import os, shutil
# Synchronous libraries that must be in a threadpool
import whisper, ollama, ssl, certifi
from lipsync import LipSync
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
from starlette.concurrency import run_in_threadpool # For running blocking code
import time
from gtts import gTTS
from google import genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()  # Load environment variables from .env file

# --- Critical SSL Fix for some environments ---
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

app = FastAPI()
security = HTTPBasic()

# --- CORS Middleware Setup ---
origins = [
    "http://localhost:8080",  # Allow requests from this frontend
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Frontend URLs allowed
    allow_credentials=True,          # Allow cookies, authorization headers
    allow_methods=["*"],             # Allow all HTTP methods
    allow_headers=["*"],             # Allow all headers
)

API_KEY = os.getenv("GEMINI_API_KEY")
try:
    print("Attempting to initialize Gemini client...")
    if API_KEY:
        # Pass the key explicitly to the client if found
        genai_client = genai.Client(api_key=API_KEY)
        print("Client initialized successfully.")
    else:
        # If API_KEY is missing, raise a clear error before calling client()
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
    
except ValueError as ve:
    # Catch the specific error raised when the environment variable is missing
    print(f"\n--- CONFIGURATION ERROR ---")
    print(f"Error: {ve}")
    print(f"ACTION REQUIRED: Please set the GEMINI_API_KEY environment variable in your terminal (e.g., export GEMINI_API_KEY='YOUR_KEY').")
    genai_client = None

uri = os.getenv("MONGO_URI")

try:
    # Create a synchronous client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB (using pymongo)!")
except Exception as e:
    # If this fails, the app cannot start or function correctly.
    print(f"CRITICAL MongoDB connection failed: {e}")
    # It's better to log the error and allow the app to try again or fail gracefully,
    # but for a working example, we proceed with the db object.
    client = None 

# Define database and collections if the client connected
if client:
    db = client["lipsync_db"]
    users_collection = db["users"]
    history_collection = db["history"]
else:
    # Use dummy objects if connection failed to prevent a crash on import, 
    # but routes will still fail.
    db = None
    users_collection = None
    history_collection = None
    

# --- Folders Setup ---
UPLOAD_FOLDER = "static"
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)




# ------------------- Models -------------------
class User(BaseModel):
    username: str
    password: str

# ------------------- Helper Functions (Synchronous) -------------------
# These functions will be called using run_in_threadpool

def db_find_one_sync(collection, query: dict):
    """Synchronous MongoDB find_one operation."""
    if collection is None:
        raise Exception("Database not connected.")
    return collection.find_one(query)

def db_insert_one_sync(collection, document: dict):
    """Synchronous MongoDB insert_one operation."""
    if collection is None:
        raise Exception("Database not connected.")
    return collection.insert_one(document)

def db_update_one_sync(collection, query: dict, update: dict):
    """Synchronous MongoDB update_one operation."""
    if collection is None:
        raise Exception("Database not connected.")
    return collection.update_one(query, update)

def db_find_one_sorted_sync(collection, query: dict, sort_criteria):
    """Synchronous MongoDB find_one operation with sorting."""
    if collection is None:
        raise Exception("Database not connected.")
    return collection.find_one(query, sort=sort_criteria)

def transcribe_audio_sync(audio_path: str) -> str:
    """Synchronous function to transcribe audio."""
    whisper_model = whisper.load_model("small")
    if not whisper_model:
        raise Exception("Whisper model not loaded.")
    result = whisper_model.transcribe(audio_path,task="transcribe",
    language="en", fp16=False)
    print(result["text"])
    
    return result['text']


def call_llm_sync(prompt: str) -> str:
    """
    Synchronous function to call the Gemini API, 
    enforcing the "Ankur at HCL" persona via system_instruction.
    
    Args:
        prompt: The user's input query.
        
    Returns:
        The generated text from the model.
    """
    if not genai_client:
        return "I don't know (Client failed to initialize)."
    system_prompt = """
    You are Ankur, an AI working at hcl.
    You are a helpful assistant that helps people find information.
    You always answer in a concise and clear manner.
    If you don't know the answer, just say "I don't know".
    Do not make up answers.
    Keep your answers brief and to the point.
    Never mention that you are an AI model.
    Use simple language that a 10 year old can understand.
    Always refer to yourself as Ankur.
    Always response in English.
    """
    
    # 2. Call the API using the google-genai SDK with system_instruction
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,  # Direct text input for user content
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt.strip()
            )
        )
        
        print("Gemini response (Ankur):")
        print(response.text)
        return response.text
        
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return "I don't know (API call failed)."



def tts_generate_sync(text: str, output_path: str):
    """Synchronous function for Text-to-Speech generation using gTTS."""

    print(f"TTS saving to path: {output_path}")
    
    try:
        # 1. Create the gTTS object
        # Using 'en' (English) and 'com' domain (default)
        tts = gTTS(text=text.strip(), lang="en", tld="com", slow=False)
        
        # 2. Save the audio directly to the specified path
        tts.save(output_path)
        
        print(f"TTS audio saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"gTTS generation failed: {e}")
        # Reraise the exception to be handled by the calling asynchronous function
        raise

def generate_video_sync(image_path: str, audio_path: str, output_path: str):
    """Synchronous function for LipSync video generation."""
    print(output_path)
    lip = LipSync(
        model='wav2lip',
        checkpoint_path='./wav2lip.pth', # Ensure this path is correct
        nosmooth=True,
        device='cpu',
        cache_dir='cache',
        img_size=96,
        save_cache=False,
    )
    lip.sync(image_path, audio_path, output_path)
    print(f"Video saved to {output_path}")
    return output_path

# ------------------- Authentication -------------------

# Authentication
async def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    print("Credentials received:", credentials.username, credentials.password)
    user = await run_in_threadpool(
        db_find_one_sync,
        users_collection,
        {"username": credentials.username}  # Query by username
    )
    print("User fetched from DB:", user)

    if user and user["password"] == credentials.password:
        return user
    raise HTTPException(status_code=401, detail="Invalid credentials")


# ------------------- Routes -------------------
@app.post("/login")
async def login(user: dict = Depends(authenticate)):
    """Authenticate user and return info"""
    image_path_internal = user.get("image_path")
    public_image_url = None
    if image_path_internal:
        relative_path = image_path_internal.lstrip('./')
        public_image_url = f"http://localhost:8000/{relative_path}"

    return JSONResponse(content={
        "message": "Login successful",
        "user_id": str(user["_id"]),
        "username": user["username"],   # Use username instead of name/email
        "image_url": public_image_url
    })

#register route


@app.post("/register")
async def register(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    image: Optional[UploadFile] = None
):
    # Check if user exists
    existing = await run_in_threadpool(
        db_find_one_sync, 
        users_collection, 
        {"username": email}
    )
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    # Save image if uploaded
    if image:
        image_path = os.path.join(IMAGE_FOLDER, f"{email}.jpg")
        content = await image.read()
        await run_in_threadpool(lambda: open(image_path, "wb").write(content))
    else:
        image_path = './Image.jpeg'  # default image

    # Insert user with name field
    await run_in_threadpool(
        db_insert_one_sync,
        users_collection,
        {
            "username": email,
            "password": password,
            "name": name,
            "image_path": image_path
        }
    )

    return {"message": "User registered successfully"}


@app.post("/upload-audio")
async def process_audio(
    audio_file: Optional[UploadFile] = File(None),
    text_input: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    user: dict = Depends(authenticate)
):
    user_id = str(user["_id"])

    

    # 2. Handle image (new upload or existing)
    if image_file:
        image_path = os.path.join(IMAGE_FOLDER, f"{user_id}.jpg")
        file_content = await image_file.read()
        await run_in_threadpool(lambda: open(image_path, "wb").write(file_content))
        
        # Update user document (pymongo in threadpool)
        await run_in_threadpool(
            db_update_one_sync, 
            users_collection, 
            {"_id": user["_id"]}, 
            {"$set": {"image_path": image_path}}
        )
    else:
        image_path = user.get("image_path")
        if not image_path:
            raise HTTPException(status_code=400, detail="No image uploaded or found in user profile.")

    # 3. Run the processing pipeline (All blocking functions in threadpool)
    try:
        # A. Save audio (file I/O is blocking, use threadpool)
        print(audio_file,'hello',text_input)
        if audio_file:
            audio_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_input.wav")
            content = await audio_file.read()
            await run_in_threadpool(lambda: open(audio_path, "wb").write(content))
            # Transcribe audio
            transcription = await run_in_threadpool(transcribe_audio_sync, audio_path)
        else:
            transcription = text_input.strip()  # Use text input as transcription
            audio_path = None  # Will generate TTS from text

        
        # B. Call LLM
        llm_response = await run_in_threadpool(call_llm_sync, transcription)
        print("Call LLM")
        
        # C. Generate TTS Audio
        tts_audio_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_tts.wav")
        await run_in_threadpool(tts_generate_sync, llm_response, tts_audio_path)
        print("Generate TTS Audio", tts_audio_path)

        max_attempts = 5
        for attempt in range(max_attempts):
            print('Checking TTS audio file...', attempt+1)
            if os.path.exists(tts_audio_path) and os.path.getsize(tts_audio_path) > 100: # Check for a reasonable size
                print(f"TTS audio verified, size: {os.path.getsize(tts_audio_path)} bytes.")
                break # Exit loop if file is good
            
            print(f"Attempt {attempt+1}: TTS file is empty or missing. Waiting...")
            time.sleep(3) # Wait half a second
        else:
            # This block executes if the loop finishes without 'break'
            if not os.path.exists(tts_audio_path):
                raise HTTPException(status_code=500, detail="TTS generation failed: output file does not exist.")
            if os.path.getsize(tts_audio_path) <= 100:
                raise HTTPException(status_code=500, detail="TTS generation failed: output file is empty (v cannot be empty).")


        # D. Generate Video
        output_video_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_output.mp4")
        await run_in_threadpool(
        generate_video_sync, image_path, tts_audio_path,output_video_path   )
        print("Generate Video")

    except Exception as e:
        print(f"Processing pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # 4. Save history (pymongo in threadpool)
    await run_in_threadpool(
        db_insert_one_sync, 
        history_collection, 
        {"user_id": user_id, "input_text": transcription, "output_text": llm_response, "video_path": output_video_path}
    )

    return {
        "transcription": transcription,
        "response": llm_response,
        "video_path": output_video_path
    }

@app.get("/get-video")
async def get_video(user: dict = Depends(authenticate)):
    user_id = str(user["_id"])
    video_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_output.mp4")
    
    if not os.path.exists(video_path):
        # Fallback check in history (pymongo in threadpool)
        last_history = await run_in_threadpool(
            db_find_one_sorted_sync, 
            history_collection, 
            {"user_id": user_id}, 
            [('_id', -1)]
        )
        if last_history and os.path.exists(last_history.get("video_path")):
            video_path = last_history["video_path"]
        else:
            raise HTTPException(status_code=404, detail="No video found for this user. Upload an audio file first.")
            
    return FileResponse(video_path, media_type="video/mp4")

