import requests
from decouple import config

ELEVEN_LABS_API_KEY = config("ELEVEN_LABS_API_KEY")

def text_to_speech(message):
    body = {
        "text": message,
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
        }
    }
    voice_emily = "CYw3kZ02Hs0563khs1Fj"

    headers = {"xi-api-key": ELEVEN_LABS_API_KEY, "Content-Type": "application/json", "accept": "audio/mpeg"}
    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_emily}"

    try:
        response = requests.post(endpoint, json=body, headers=headers)
    except Exception as e:
        print(f"There was an error {e}")

    if response.status_code==200:
        return response.content