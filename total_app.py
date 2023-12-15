import streamlit as st
import cv2
import time
from reponse_generation import generate_response, voice_generation
from demo import InferencePipeline
from hydra import compose, initialize
import numpy as np

def click_start():
    st.session_state.start = True

def click_stop():
    st.session_state.stop = True
    st.session_state.start = False
    st.session_state.record_in_progress = False
    st.session_state.record_in_progress_1 = False
    st.session_state.record_number = 1

def click_record():
    st.session_state.record_in_progress = True
    st.session_state.record_in_progress_1 = True
def click_stop_record():
    st.session_state.record_in_progress_1 = False
    # print('yes')


# @hydra.main(version_base="1.3", config_path="configs", config_name="config")
def all_models_launch(frame_placeholder, file_name, recording_duration=5):

    cap = cv2.VideoCapture(0)
    stop_button_frame = st.empty()
    stop_button_pressed = stop_button_frame.button('Stop recording the video', on_click=click_stop_record)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(f'videos/video_of_{file_name}_{st.session_state.record_number}.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))
    start_time = time.time()
    elapsed_time = 0

    while elapsed_time < recording_duration and st.session_state.record_in_progress_1:
        ret, frame = cap.read()
        if not ret:
            st.write("The recording ended")
            break


        writer.write(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels='RGB')
        elapsed_time = time.time() - start_time

        if stop_button_pressed:
            writer.release()

    st.session_state.record_in_progress_1 = False


    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    time.sleep(3)

    stop_button_frame.write("The recording ended. Preprocessing video...")
    with initialize(version_base="1.3", config_path="configs"):
        cfg = compose(config_name="config", overrides=["data.modality=video", f'file_path=videos/video_of_{file_name}_{st.session_state.record_number}.mp4'])
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.file_path).lower()

    stop_button_frame.write(f':blue[Your phrase was: {transcript}]')

    answer = generate_response(transcript + ' ?')

    response_frame = st.empty()
    response_frame.write(f':red[Generated answer: {answer}]')

    generated_audio = voice_generation(answer)

    audio_file = open('response/speech.wav', 'rb')
    audio_bytes = audio_file.read()

    generated_audio_frame = st.empty()
    col3, col4 = generated_audio_frame.columns(2)
    with col3:
        st.text('Generated audio response')
    with col4:
        st.audio(audio_bytes, format='audio/wav')

    st.session_state.record_number += 1

    st.session_state.record_in_progress=False

def main_content():
    st.title(':rainbow[Final Project: Lip Reader. INF573 - Computer Vision (2023-2024)]')
    st.divider()

    input_frame = st.empty()
    file_name = input_frame.text_input("Enter your name, please 🙏🏻", key='unique_key')

    title_frame = st.empty()


    button_frame = st.empty()
    if file_name:
        title_frame.title("Video recording")
        col1, col2 = button_frame.columns(2)
        with col1:
            start_button = st.button('Start application', on_click=click_start)
        with col2:
            end_button = st.button('Shut down application', on_click=click_stop)

        temp = 1
        if st.session_state.start:
            record_frame = st.empty()
            record_button = record_frame.button(f'Record video', on_click=click_record)
            if st.session_state.record_in_progress:
                frame_camera = st.empty()
                all_models_launch(frame_camera, file_name)

        if end_button:
            button_frame.empty()
            input_frame.empty()
            title_frame.empty()
            warning_frame = st.empty()
            warning_frame.warning('Thank you so much! See you soon❤️', icon='🥳')
            restart_frame = st.empty()
            restart_button = restart_frame.button("Restart application")

            if restart_button:
                restart_frame.empty()
                main_content()

def fixed_content():

    # Set page title and favicon
    st.set_page_config(page_title="Final Project: Lip Reader. INF573-Computer Vision (2023-2024)", page_icon="🔥")

    # Sidebar
    st.sidebar.subheader("Project description ")
    st.sidebar.markdown("> An application that allows users to have video conversations with a virtual interlocutor. "
            "Users record videos, and our system processes them without audio (only lip's movements, however it's possible to add audio as well), "
            "identifying speaker's phrase or question using META's new model `Auto-AVSR`. "
            "Speech recognition extracts phrases, forming textual responses through language models. "
            "The system then generates a natural voice response using advanced speech synthesis. ")
    st.sidebar.subheader("Group: Tarasov Aleksandr, Sellage Kulakshi Bhashini Fernando (p24)")
    st.sidebar.image('Aleksandr.png', caption='Aleksandr', use_column_width=True)
    st.sidebar.image("Kulakshi.png", caption="Kulakshi", use_column_width=True)


if __name__ == '__main__':

    if 'start' not in st.session_state:
        st.session_state.start = False
    if 'stop' not in st.session_state:
        st.session_state.stop = False
    if 'record_number' not in st.session_state:
        st.session_state.record_number = 1
    if 'record_in_progress' not in st.session_state:
        st.session_state.record_in_progress = False
    if 'record_in_progress_1' not in st.session_state:
        st.session_state.record_in_progress_1 = False

    fixed_content()
    main_content()









