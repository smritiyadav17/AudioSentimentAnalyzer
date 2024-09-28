import gradio as gr
from src.audio_sentiment_analysis import AudioSentimentAnalysisApp
from src.speech_to_text import SpeechToText
from src.sentiment_analyzer import SentimentAnalyzer


def launch_app():
    speech_to_text = SpeechToText()
    sentiment_analyzer = SentimentAnalyzer()
    app = AudioSentimentAnalysisApp(speech_to_text, sentiment_analyzer)

    iface = gr.Interface(
        fn=app.process_audio,
        inputs=gr.Audio(type="filepath", label="Upload your audio file 🎧"),
        outputs=gr.Textbox(label="Sentiment Output 📊"),
        title="Audio Sentiment Analysis",
        description="Upload an audio file, and this app will transcribe it and analyze the sentiment.",
    )
    iface.launch()


if __name__ == "__main__":
    launch_app()
