import openai
from decouple import config
from functions.chat_history import get_recent_messages
openai.api_key = config("OPENAI_API_KEY")


#OPENAI - Whisper

def audio_to_text(audio_file):
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        message_text = transcript["text"]
        return message_text
    except Exception as e:
        print(f"There was an error: {e}")

def get_completion_from_audio(converted_audio, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0.5, 
                                 max_tokens=50):
    messages = get_recent_messages()
    messages.append({
        "role":"user", "content": converted_audio
    })
    print(messages)
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
            max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
        )
        print(response)
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"There was an error: {e}")
        return
