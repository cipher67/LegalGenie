import streamlit as st
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random
import os
from backend import process_question
from openai import OpenAI

# Initialize OpenAI client
openai_api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def stt_transcribe(audio_file):
    """Function to transcribe audio to text using OpenAI's Whisper model."""
    with open(audio_file, "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio),
        )
        return transcript.text

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="LegalGenie",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>LegalGenie</i> 💬</h1>""")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the previous messages if there are any
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"])
                elif content["type"] == "video_file":
                    st.video(content["video_file"])
                elif content["type"] == "audio_file":
                    st.audio(content["audio_file"])

    with st.sidebar:
        # Image Upload
        st.write(f"### **🖼️ Add an image or a video file:**")

        def add_image_to_messages():
            if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                if img_type == "video/mp4":
                    video_id = random.randint(100000, 999999)
                    with open(f"video_{video_id}.mp4", "wb") as f:
                        f.write(st.session_state.uploaded_img.read())
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": [{
                                "type": "video_file",
                                "video_file": f"video_{video_id}.mp4",
                            }]
                        }
                    )
                else:
                    raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                    img = get_image_base64(raw_img)
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"}
                            }]
                        }
                    )

        cols_img = st.columns(2)

        with cols_img[0]:
            with st.popover("📁 Upload"):
                st.file_uploader(
                    "Upload an image or a video:",
                    type=["png", "jpg", "jpeg", "mp4"],
                    accept_multiple_files=False,
                    key="uploaded_img",
                    on_change=add_image_to_messages,
                )

        with cols_img[1]:
            with st.popover("📸 Camera"):
                activate_camera = st.checkbox("Activate camera (only images)")
                if activate_camera:
                    st.camera_input(
                        "Take a picture",
                        key="camera_img",
                        on_change=add_image_to_messages,
                    )

        # Audio Upload
        st.write("#")
        st.write(f"### **🎤 Add an audio (Speech To Text):**")

        audio_prompt = None
        audio_file_added = False
        if "prev_speech_hash" not in st.session_state:
            st.session_state.prev_speech_hash = None

        speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
        if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
            st.session_state.prev_speech_hash = hash(speech_input)
            # Save the audio file
            audio_id = random.randint(100000, 999999)
            audio_file = f"audio_{audio_id}.wav"
            with open(audio_file, "wb") as f:
                f.write(speech_input)

            # Transcribe the audio file
            audio_prompt = stt_transcribe(audio_file)

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [{
                        "type": "audio_file",
                        "audio_file": audio_file,
                    }]
                }
            )
            audio_file_added = True

        st.divider()
        def reset_conversation():
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                st.session_state.pop("messages", None)

        st.button(
            "🗑️ Reset conversation",
            on_click=reset_conversation,
        )

    # Chat input
    prompt = st.chat_input("Hi! Ask me anything...")

    if prompt or audio_prompt:
        user_input = prompt or audio_prompt

        st.session_state.messages.append(
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": user_input,
                }]
            }
        )

        # Display the new messages
        with st.chat_message("user"):
            st.markdown(user_input)

        # Send the question to the backend and get the response
        response = process_question(user_input)

        # Display the response
        with st.chat_message("assistant"):
            st.write(response)

        # Add the response to the conversation history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": response,
                }]
            }
        )

if __name__ == "__main__":
    main()
