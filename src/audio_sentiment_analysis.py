from src.speech_to_text import SpeechToText
from src.sentiment_analyzer import SentimentAnalyzer


class AudioSentimentAnalysisApp:
    def __init__(
        self, speech_to_text: SpeechToText, sentiment_analyzer: SentimentAnalyzer
    ):
        self.speech_to_text = speech_to_text
        self.sentiment_analyzer = sentiment_analyzer

    def process_audio(self, audio):
        transcription = self.speech_to_text.transcribe(audio)
        return self.sentiment_analyzer.analyze(transcription[0])
