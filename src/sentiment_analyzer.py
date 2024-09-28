import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "tabularisai/robust-sentiment-analysis"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sentiment_map = {
            0: "Very Negative",
            1: "Negative",
            2: "Neutral",
            3: "Positive",
            4: "Very Positive",
        }

    def analyze(self, text):
        inputs = self.tokenizer(
            text.lower(),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return self.sentiment_map[predicted_class]
