# 🧠 AI Avatar Chat – Voice-to-Avatar Conversational System

An interactive **AI avatar web application** that listens to user voice, understands the message using **Whisper ASR**, generates intelligent text replies using **Gemini AI**, and speaks back using **gtts** voice synthesis.  
The system also generates a **lip-synced avatar video** that visually responds to the user.

---

## 🚀 Features

✅ Voice-to-Text via **Whisper ASR**  
✅ Text Intelligence using **Gemini Live API**  
✅ Realistic Text-to-Speech via **GTTS**  
✅ Lip-Synced Avatar Response via **LipSync**  
✅ Secure Login & Registration (FastAPI + MongoDB)  
✅ Modern Frontend built with **React + TypeScript**  
✅ Full Audio/Video Interaction Cycle  
✅ CORS-enabled backend (`http://localhost:8080`)  

---

## 🏗️ Project Architecture

```plaintext
🎤 User speaks
   ↓
🎧 Frontend (React)
   - Records voice (Web Audio API)
   - Sends to FastAPI backend with credentials
   ↓
⚙️ FastAPI Backend
   - Authenticates user (HTTPBasic)
   - Transcribes audio (Whisper)
   - Generates response (Gemini API)
   - Converts to speech (gtts)
   - Creates lip-sync video (LipSync)
   ↓
🧑‍💻 MongoDB
   - Stores user credentials & avatars
   ↓
🎬 Frontend
   - Displays avatar video
   - Shows chat history and messages
🧩 Technologies Used
Component	Technology
Frontend	React + TypeScript + TailwindCSS
Backend	FastAPI (Python)
Database	MongoDB (pymongo)
AI Text Model	Gemini Live API
Speech-to-Text	Whisper
Text-to-Speech	gtts
Avatar Lip Sync	LipSync
Authentication	HTTPBasic & MongoDB
Deployment	Uvicorn / Render / Vercel (optional)

🛠️ Installation Guide
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/your-username/ai-avatar-chat.git
cd ai-avatar-chat
2️⃣ Backend Setup (FastAPI)
🐍 Create Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
📦 Install Dependencies
bash
Copy code
pip install fastapi uvicorn pymongo whisper pyttsx3 gtts google-generativeai lipsync certifi python-multipart
⚙️ Set Environment Variables
Create a .env file inside your backend directory:

bash
Copy code
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_gemini_api_key
MONGO_URI=your_mongodb_connection_string
▶️ Run FastAPI Server
bash
Copy code
uvicorn main:app --reload
Backend runs on:
👉 http://localhost:8000

3️⃣ Frontend Setup (React + Vite or Next.js)
bash
Copy code
cd frontend
npm install
npm run dev
Frontend runs on:
👉 http://localhost:8080

🔗 API Endpoints
Endpoint	Method	Description
/register	POST	Register a new user (FormData: username, password, image)
/upload-audio	POST	Upload recorded audio and get AI response
/get-video	GET	Fetch the latest avatar response video
/login	POST	(Optional) Login endpoint for authentication


💬 Example Flow
User registers with a username, password, and optional image.

Frontend records audio and sends it to /upload-audio.

FastAPI:

Transcribes with Whisper

Generates text with Gemini API

Converts text to speech using gtts

Creates a lip-synced avatar video

Frontend displays the avatar and the AI’s spoken response.

🔐 Authentication
Uses HTTP Basic Auth for secure communication between frontend and backend.

User credentials are stored in MongoDB.

Each request (like /upload-audio) must include:

js
Copy code
Authorization: Basic base64(username:password)
🧠 Gemini AI Integration (Text Generation)
python
Copy code
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def call_llm_sync(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
🔊 gtts Integration
python
Copy code
from quick_tts import text_to_speech

def generate_tts_response(text: str, output_path: str = "static/output.wav"):
    text_to_speech(
        text=text,
        output_file=output_path,
        model="tts-1-hd",
        voice="nova"
    )
    return output_path
🎨 Frontend Components Overview
Component	Purpose
AuthModal.tsx	Handles signup/login
AvatarDisplay.tsx	Displays avatar & video
ChatInterface.tsx	Chat message interface
VoiceRecorder.tsx	Records and sends audio
index.tsx	Main app controller

🧩 CORS Configuration
Enabled in backend to allow frontend:

python
Copy code
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
🧾 Folder Structure
arduino
Copy code
📦 ai-avatar-chat
├── backend/
│   ├── main.py
│   ├── static/
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
└── README.md
🧪 Testing
Run Backend Tests
bash
Copy code
pytest
Run Frontend in Development
bash
Copy code
npm run dev
🚢 Deployment
You can deploy:

Backend  AWS EC2

Frontend S3

Make sure to set environment variables in your host settings.

🧑‍💻 Author
Ankur Yadav
AI-DS Engineer (React + Django + FastAPI)

🌐 GitHub: https://github.com/ankur-06-2003

📜 License
This project is licensed under the MIT License — you’re free to use, modify, and distribute it.

❤️ Acknowledgments
FastAPI

Google Gemini API

gtts

Whisper

LipSync

yaml
Copy code

---

Would you like me to include **architecture diagram (flowchart in Markdown or PNG)** inside this README (so it visually shows the data flow between React → FastAPI → Gemini → TTS → LipSync → Frontend)?






according to file





ChatGPT can make mistakes. Check important info. See Cookie Preferences.
