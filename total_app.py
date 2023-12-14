import streamlit as st
import cv2


def video_shooting_st(stop_button_pressed, frame_placeholder, file_name):
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(f'videos/video_of_{file_name}.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))

    while not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.write("The recording ended")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels='RGB')

        if cv2.waitKey(1) & 0xFF == 27 or stop_button_pressed:
            st.write("The recording ended")
            break


        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


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
            start_button = st.button('Start application')
            # print(start_button)
        with col2:
            end_button = st.button('Shut down application')

        if start_button:
            stop_button_pressed = st.button('Stop recording the video')
            frame_camera = st.empty()
            video_shooting_st(stop_button_pressed, frame_camera, file_name)

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
    fixed_content()
    main_content()









