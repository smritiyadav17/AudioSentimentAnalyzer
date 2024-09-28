import unittest
from src.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def test_analyze_sentiment(self):
        test_text = "I love this product!"
        sentiment = self.sentiment_analyzer.analyze(test_text)
        self.assertIn(
            sentiment,
            ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
        )


if __name__ == "__main__":
    unittest.main()
