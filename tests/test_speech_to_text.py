import unittest
import os
from src.speech_to_text import SpeechToText


class TestSpeechToText(unittest.TestCase):

    def setUp(self):
        self.speech_to_text = SpeechToText()

    def test_transcribe_audio(self):
        audio_file_path = os.path.join(
            os.path.dirname(__file__), "sample_audio_files", "harvard.wav"
        )
        # Mock audio file path (replace with an actual audio file for real tests)
        # audio_file_path = "sample_audio_files/harvard.wav"
        transcription = self.speech_to_text.transcribe(audio_file_path)[0]
        self.assertIsInstance(transcription, str)  # Check if transcription is a string
        self.assertGreater(len(transcription), 0)  # Ensure transcription is not empty


if __name__ == "__main__":
    unittest.main()
