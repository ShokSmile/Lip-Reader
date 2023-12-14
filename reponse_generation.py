from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

MODEL_NAME = 'google/t5-small-ssm-nq'
SPEECH_GENERATION_MODEL = "microsoft/speecht5_tts"


def generate_response(question: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenized_question = tokenizer(question, return_tensors='pt').input_ids

    tokenized_answer = model.generate(tokenized_question, max_new_tokens=50)[0]

    answer = tokenizer.decode(tokenized_answer, skip_special_tokens=True)

    print(f"Question: {question}\nResponse: {answer}")
    return answer


def voice_generation(answer: str):
    processor = SpeechT5Processor.from_pretrained(SPEECH_GENERATION_MODEL)
    model = SpeechT5ForTextToSpeech.from_pretrained(SPEECH_GENERATION_MODEL)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=answer, return_tensors="pt")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("response/speech.wav", speech.numpy(), samplerate=16000)


if __name__ == '__main__':
    answer = generate_response('How are you?')

    voice_generation(answer)