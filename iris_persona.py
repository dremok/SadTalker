import json
import os
import sys
from pathlib import Path

import gradio as gr
import requests
from elevenlabs import set_api_key, generate, Voice

from iris_utils import timeit
# Read environment variables
from predict import Predictor

elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
set_api_key(elevenlabs_api_key)

AUTHORIZATION_HEADER = os.getenv('AUTHORIZATION_HEADER')


@timeit
def get_persona_id(base_url, headers, persona_name):
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        persona_id = [p["id"] for p in response.json()["personas"] if p["name"] == persona_name][0]
        return persona_id
    else:
        print("Failed to retrieve personas. Status code:", response.status_code)
        sys.exit(1)


@timeit
def get_access_key(base_url, headers, persona_id):
    chat_url = f"{base_url}/{persona_id}/chat_sessions"
    response = requests.post(chat_url, headers=headers)
    if response.status_code == 200 or response.status_code == 201:
        new_chat = response.json()["chat_session"]
        access_key = new_chat["access_key"]
    else:
        print("Failed to create chat session. Status code:", response.status_code)
        sys.exit(1)
    return access_key


@timeit
def get_message(headers, access_key, message):
    print(message)
    print(f"access_key={access_key}")
    data = {
        "message": message,
        "context": ["Always respond briefly, use maximum 5 words in each response."]
    }
    chat_url = f"https://api.irisona.net/v0/chat_sessions/{access_key}/messages"
    response = requests.post(chat_url, headers=headers, json=data, stream=True)
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                json_response = json.loads(decoded_line)
                if json_response.get('type') == 'message':
                    content = json_response["content"]
                    print(content)
                    return content
    else:
        print("Failed to send message. Status code:", response.status_code)


@timeit
def generate_audio(message, out_file):
    audio = generate(
        text=message,
        voice=Voice(
            voice_id='O2Zd2yBa2Mt5cZIhfYW5',
        )
    )
    with open(out_file, "wb") as file:
        file.write(audio)


@timeit
def generate_avatar(audio_file, predictor) -> str:
    return predictor.predict(Path("image00009.jpeg"),
                             Path(audio_file),
                             enhancer=None,
                             preprocess="crop",
                             ref_eyeblink=None,
                             ref_pose=None,
                             still=True)


headers = {'Authorization': os.getenv('AUTHORIZATION_HEADER'), 'Accept': 'application/json'}
base_url = "https://api.irisona.net/v0/personas"
predictor = Predictor()
predictor.setup()

persona_id = get_persona_id(base_url, headers, persona_name="Arvidai")
access_key = get_access_key(base_url, headers, persona_id)


def chat_and_generate_video(user_message):
    """
    This function takes user input, gets a response from the chatbot,
    generates audio, and creates an avatar video.
    """

    # Get chatbot response
    message = get_message(headers, access_key, user_message)

    # Assuming generate_audio and generate_avatar are adapted to work with your current setup
    audio_file = "output_audio.mp3"
    generate_audio(message, audio_file)
    video_path = generate_avatar(audio_file, predictor)  # Ensure this returns an MP4 file path

    return video_path


iface = gr.Interface(
    fn=chat_and_generate_video,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message to Arvida..."),
    outputs=gr.Video(),
    title="Chat with Arvida",
    description="Enter your message and get a response from Arvida, along with a generated video response.",
)

if __name__ == "__main__":
    iface.launch()
