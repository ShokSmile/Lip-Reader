import os
import torch
import torchaudio
import torchvision
from lightning import ModelModule
from datamodule.transforms import AudioTransform, VideoTransform
from hydra import compose, initialize



class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="mediapipe"):
        super(InferencePipeline, self).__init__()
        self.modality = 'video'
        # if self.modality in ["audio", "audiovisual"]:
        #     self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
                self.video_transform = VideoTransform(subset="test")
            # elif detector == "retinaface":
            #     from preparation.detectors.retinaface.detector import LandmarksDetector
            #     from preparation.detectors.retinaface.video_process import VideoProcess
            #     self.landmarks_detector = LandmarksDetector(device="cuda:0")
            #     self.video_process = VideoProcess(convert_gray=False)

            else:
                raise ValueError("Use only mediapipe, please")

            cfg.data.modality = 'video'
            self.modelmodule = ModelModule(cfg)
            self.modelmodule.model.load_state_dict(
                torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
            self.modelmodule.eval()

        else:
            raise ValueError('Please provide modality from {audio, audiovisual}')

    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        # if self.modality == "audio":
        #     audio, sample_rate = self.load_audio(data_filename)
        #     audio = self.audio_process(audio, sample_rate)
        #     audio = audio.transpose(1, 0)
        #     audio = self.audio_transform(audio)
        #     with torch.no_grad():
        #         transcript = self.modelmodule(audio)

        if self.modality == "video":
            video = self.load_video(data_filename)
            landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)
            with torch.no_grad():
                transcript = self.modelmodule(video)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


# @hydra.main(config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.file_path)
    print(f"transcript: {transcript}")
    return transcript



if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="configs"):
        cfg = compose(config_name="config", overrides=["data.modality=video", f'file_path=videos/video_of_aleksandr.mp4'])
    main(cfg)
