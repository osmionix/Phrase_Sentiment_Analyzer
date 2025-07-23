import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

class SentimentModel:
    def __init__(self):
        self.pipeline = joblib.load('models/classifier.pkl')
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def predict(self, text):
        processed_text = self.preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])
        return "Positive" if prediction[0] == 1 else "Negative"