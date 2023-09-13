#uvicorn app:app
#uvicorn app:app --reload

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse #streaming response for text to speech
from fastapi.middleware.cors import CORSMiddleware # front end backend resource sharing
from decouple import config # bring environment variables in 

from functions.openai_requests import audio_to_text, get_completion_from_audio
from functions.chat_history import store_messages, new_chat
from functions.text_to_speech import text_to_speech

app = FastAPI()

# CORS - Origins 
origins = ["http://localhost:5173",
           "http://localhost:5174",
           "http://localhost:4174",
           "http://localhost:3000"] # where requests can originate from (frontend requests backend)

# CORS - Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # all types (GET, POST, PUT, etc)
    allow_headers=["*"]

)

@app.get("/health")
async def root():
    return {"message": "healthy"}

@app.get("/new_chat")
async def create_new_chat():
    new_chat()
    return {"message": "new chat created"}

#Get audio from file
#Note: audio won't play in browser when doing post
@app.post("/post-audio/")
async def post_audio(file: UploadFile = File(...)):
    
    #save frontend file
    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    #open saved file
    audio_input = open(file.filename, "rb")

    text = audio_to_text(audio_input)

    if not text:
        return HTTPException(status_code=400, detail="Failed to decode audio")
    
    chat_response=get_completion_from_audio(text)

    if not chat_response:
        return HTTPException(status_code=400, detail="Failed to get chat reponse")

    store_messages(text, chat_response)

    audio_output = text_to_speech(chat_response)

    if not audio_input:
        return HTTPException(status_code=400, detail="Failed to get Eleven Labs audio reponse")
    
    def iterfile():
        yield audio_output
    
    return StreamingResponse(iterfile(), media_type="application/octet-stream")
