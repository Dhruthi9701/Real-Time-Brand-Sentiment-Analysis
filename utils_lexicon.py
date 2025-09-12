# utils_lexicon.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)
_s = SentimentIntensityAnalyzer()

def vader_scores(text):
    if text is None:
        text = ''
    return _s.polarity_scores(str(text))
