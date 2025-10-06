import numpy as np
import pandas as pd
import re
import nltk
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = None
        self.tfidf = None
        self.ps = PorterStemmer()
        self.model_path = 'models/'

        # Label mapping: numeric to string
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.reverse_label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        # Ensure models directory exists
        os.makedirs(self.model_path, exist_ok=True)

        # Download NLTK data if not already present
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def preprocess_text(self, text):
        """Preprocess a single text string"""
        if not isinstance(text, str):
            return ""

        # Remove non-alphabetic characters
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()

        # Remove stopwords (except 'not')
        all_stopwords = stopwords.words('english')
        if 'not' in all_stopwords:
            all_stopwords.remove('not')

        # Stem words
        text = [self.ps.stem(word) for word in text if word not in set(all_stopwords)]

        return ' '.join(text)

    def train_model(self, csv_path='Twitter_Data.csv'):
        """Train the sentiment analysis model"""
        print("Loading dataset...")

        # Create sample data if CSV doesn't exist
        if not os.path.exists(csv_path):
            print("Creating sample dataset...")
            self.create_sample_data(csv_path)

        dataset = pd.read_csv(csv_path)
        dataset = dataset.dropna(subset=['category']).reset_index(drop=True)

        print("Preprocessing text data...")
        corpus = []
        for text in dataset['clean_text']:
            if isinstance(text, str):
                processed_text = self.preprocess_text(text)
                corpus.append(processed_text)
            else:
                corpus.append("")

        print("Vectorizing text...")
        self.tfidf = TfidfVectorizer(max_features=10000, min_df=3, ngram_range=(1,2))
        X = self.tfidf.fit_transform(corpus)

        # Handle different category formats
        if dataset['category'].dtype == 'object':
            # String categories - map to numeric
            y = dataset['category'].map(self.reverse_label_map).values
        else:
            # Numeric categories - map -1,0,1 to 0,1,2
            y = dataset['category'].map({-1: 0, 0: 1, 1: 2}).values

        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Logistic Regression model...")
        self.classifier = LogisticRegression(max_iter=300, C=2.0, solver='lbfgs')
        self.classifier.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🔹 Logistic Regression Results 🔹")
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                  target_names=['negative', 'neutral', 'positive']))

        # Save model and vectorizer
        self.save_model()

        return accuracy

    def save_model(self):
        """Save the trained model and vectorizer"""
        joblib.dump(self.classifier, os.path.join(self.model_path, 'sentiment_model.pkl'))
        joblib.dump(self.tfidf, os.path.join(self.model_path, 'tfidf_vectorizer.pkl'))
        print("Model and vectorizer saved successfully!")

    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            self.classifier = joblib.load(os.path.join(self.model_path, 'sentiment_model.pkl'))
            self.tfidf = joblib.load(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'))
            print("Model and vectorizer loaded successfully!")
            return True
        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
            return False

    def predict_sentiment(self, text):
        """Predict sentiment for a given text"""
        if self.classifier is None or self.tfidf is None:
            if not self.load_model():
                return {"error": "Model not trained. Please train the model first."}

        # Preprocess the input text
        processed_text = self.preprocess_text(text)

        if not processed_text.strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "probabilities": {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            }

        # Vectorize the text
        text_vector = self.tfidf.transform([processed_text])

        # Predict sentiment
        prediction_numeric = self.classifier.predict(text_vector)[0]
        probabilities = self.classifier.predict_proba(text_vector)[0]

        # Convert numeric prediction to string
        prediction = self.label_map[prediction_numeric]

        # Create probability dictionary with string labels
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            label = self.label_map[i]
            prob_dict[label] = float(prob)

        # Get confidence (highest probability)
        confidence = float(max(probabilities))

        return prediction, confidence, prob_dict

    def create_sample_data(self, csv_path):
        """Create sample data for demonstration with numeric categories"""
        sample_data = {
            'clean_text': [
                "I love this product it's amazing and works perfectly",
                "This is terrible I hate it completely disappointed",
                "It's okay nothing special but does the job fine",
                "Absolutely fantastic experience exceeded all my expectations",
                "Worst purchase ever made total waste of money",
                "Pretty good overall satisfied with the quality",
                "I'm so happy with this purchase highly recommend",
                "Not satisfied at all poor quality and service",
                "Average quality product nothing to complain about",
                "Excellent service and outstanding quality love it",
                "Disappointing results not what I expected at all",
                "Neutral opinion about this product it's fine",
                "Outstanding performance and incredible value for money",
                "Poor customer service and terrible experience",
                "Decent value for money good enough for the price",
                "Love all the features and functionality amazing",
                "Complete waste of time and money avoid this",
                "It's fine I guess nothing special about it",
                "Incredible quality and fantastic customer support",
                "Completely useless and poorly designed product",
                "Great product works as expected very satisfied",
                "Bad experience would not recommend to anyone",
                "Okay product meets basic requirements nothing more",
                "Superb quality and excellent performance love it",
                "Horrible experience worst customer service ever",
                "Good enough for the price decent quality",
                "Amazing features and wonderful user experience",
                "Terrible quality broke after one day",
                "Standard product does what it's supposed to do",
                "Fantastic value and incredible performance"
            ],
            'category': [
                2, 0, 1, 2, 0, 2, 2, 0, 1, 2,  # First 10
                0, 1, 2, 0, 1, 2, 0, 1, 2, 0,  # Second 10
                2, 0, 1, 2, 0, 1, 2, 0, 1, 2   # Last 10
            ]
        }

        df = pd.DataFrame(sample_data)
        df.to_csv(csv_path, index=False)
        print(f"Sample dataset created at {csv_path}")
        print("Category mapping: 0=negative, 1=neutral, 2=positive")

    def get_model_info(self):
        """Get information about the current model"""
        if self.classifier is None:
            return {
                "model_loaded": False,
                "message": "No trained model available"
            }

        return {
            "model_loaded": True,
            "model_type": "Logistic Regression",
            "model_params": {
                "max_iter": 300,
                "C": 2.0,
                "solver": "lbfgs"
            },
            "vectorizer_params": {
                "max_features": 10000,
                "min_df": 3,
                "ngram_range": "(1,2)"
            },
            "classes": list(self.label_map.values()),
            "label_mapping": self.label_map
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.train_model()
