import torch
import librosa
from transformers import (
    Speech2TextProcessor,
    Speech2TextForConditionalGeneration,
)


class SpeechToText:
    def __init__(self):
        self.model = Speech2TextForConditionalGeneration.from_pretrained(
            "facebook/s2t-small-librispeech-asr"
        )
        self.processor = Speech2TextProcessor.from_pretrained(
            "facebook/s2t-small-librispeech-asr"
        )

    def load_audio(self, audio_file_path):
        audio, sampling_rate = librosa.load(audio_file_path, sr=16000)
        return audio, sampling_rate

    def transcribe(self, audio):
        audio, sampling_rate = self.load_audio(audio)
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"], attention_mask=inputs["attention_mask"]
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
