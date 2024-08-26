import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random

dotenv.load_dotenv()

def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]

    return gemini_messages

# Function to query and stream the response from the LLM
def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""

    if model_type == "openai":
        client = OpenAI(api_key=api_key)
        for chunk in client.chat.completions.create(
            model=model_params["model"] if "model" in model_params else "gpt-4o",
            messages=st.session_state.messages,
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
            stream=True,
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name = model_params["model"],
            generation_config={
                "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
            }
        )
        gemini_messages = messages_to_gemini(st.session_state.messages)
        print("st_messages:", st.session_state.messages)
        print("gemini_messages:", gemini_messages)
        for chunk in model.generate_content(gemini_messages):
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

    st.session_state.messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})


# Function to convert file to base64
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

    # --- Main Content ---
    openai_api_key = os.getenv("OPEN_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    client = OpenAI(api_key=openai_api_key)

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

        model_params = {
            "model": os.getenv("MODEL"),
            "temperature": 0.3,
        }

        model = os.getenv("MODEL")
        model_type = os.getenv("MODEL_TYPE")

        # Image Upload
        if model in ["gpt-4o", "gpt-4-turbo", "gemini-1.5-flash", "gemini-1.5-pro"]:
            st.write(f"### **🖼️ Add an image{' or a video file' if model_type == 'google' else ''}:**")

            def add_image_to_messages():
                if st.session_state.uploaded_img or (
                        "camera_img" in st.session_state and st.session_state.camera_img):
                    img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                    if img_type == "video/mp4":
                        # save the video file
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
                        f"Upload an image{' or a video' if model_type == 'google' else ''}:",
                        type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []),
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
        st.write(f"### **🎤 Add an audio{' (Speech To Text)' if model_type == 'openai' else ''}:**")

        audio_prompt = None
        audio_file_added = False
        if "prev_speech_hash" not in st.session_state:
            st.session_state.prev_speech_hash = None

        speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
        if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
            st.session_state.prev_speech_hash = hash(speech_input)
            if model_type == "openai":
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", speech_input),
                )

                audio_prompt = transcript.text

            elif model_type == "google":
                # save the audio file
                audio_id = random.randint(100000, 999999)
                with open(f"audio_{audio_id}.wav", "wb") as f:
                    f.write(speech_input)

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": [{
                            "type": "audio_file",
                            "audio_file": f"audio_{audio_id}.wav",
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
    if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt or audio_file_added:
        if not audio_file_added:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt or audio_prompt,
                    }]
                }
            )

            # Display the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

        else:
            # Display the audio file
            with st.chat_message("user"):
                st.audio(f"audio_{audio_id}.wav")

        with st.chat_message("assistant"):
            st.write_stream(
                stream_llm_response(
                    model_params=model_params,
                    model_type=model_type,
                    api_key=openai_api_key if model_type == "openai" else google_api_key)
            )


if __name__=="__main__":
    main()
