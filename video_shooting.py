import cv2

def shoot_video(file_name: str):
    # file_name = input("Please, provide your name: ")

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(f'videos/video_of_{file_name}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    while True:
        ret, frame = cap.read()

        writer.write(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
   # shoot_video('a')
   from hydra import compose, initialize
   from omegaconf import OmegaConf

   with initialize(version_base="1.3",config_path="configs"):
       cfg = compose(config_name="config", overrides=["data.modality=video", f"file_path=videos/video_of_{5}_{8}.mp4"])

   print(OmegaConf.to_yaml(cfg))


