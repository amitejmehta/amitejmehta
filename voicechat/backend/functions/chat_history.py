import json
import random

def get_recent_messages():
    file_name="chat_history.json"
    learn_instruction = {
        "role": "system",
        "content": """You are the head of Sales Engineering at Groove, named Tyler, the sales engagement platform. You are demoing the Groove platform to me and my job is to ask questions. """
    }

    messages=[]
    """x=random.uniform(0, 1)
    if x < 0.5:
        learn_instruction["content"] = learn_instruction["content"] + " Your response will include a comparison of Groove to its competitors. "
    else:
        learn_instruction["content"] = learn_instruction["content"] + " Your reponse will extremely inquisitive, asking details about the features I demo."""
    messages.append(learn_instruction)

    try:
        with open(file_name) as f:
            data = json.load(f)

        if data:
            for item in data:
                messages.append(item)
    except Exception as e:
        print(f"There was an error {e}")
        pass
    
    return messages

def store_messages(user_message, assistant_reponse):
    file_name = "chat_history.json"

    messages = get_recent_messages()[1:]
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_reponse})

    with open(file_name, "w") as f:
        json.dump(messages, f)

def new_chat():
    open("chat_history.json", "w")

