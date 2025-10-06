from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from model import SentimentAnalyzer

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS with more permissive settings for development
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=cors_origins, supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'OPTIONS'])

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

# Load model on application startup
print("Loading sentiment analysis model...")
if not analyzer.load_model():
    # This part is crucial for Render. If you have a pre-trained model,
    # make sure it's committed to your repository so it can be loaded.
    print("No pre-trained model found. Training a new model...")
    analyzer.train_model()

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "Sentiment Analysis API is running!",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze sentiment of text",
            "GET /health": "Health check",
            "POST /train": "Train the model (development only)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_sentiment():
    """Analyze sentiment of provided text"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        print(f"Received request: {request.method}")
        print(f"Request headers: {dict(request.headers)}")

        data = request.get_json()
        print(f"Request data: {data}")

        if not data or 'text' not in data:
            error_response = {
                "success": False,
                "error": "Missing 'text' field in request body"
            }
            print(f"Error response: {error_response}")
            return jsonify(error_response), 400

        text = data['text'].strip()

        if not text:
            error_response = {
                "success": False,
                "error": "Text cannot be empty"
            }
            print(f"Error response: {error_response}")
            return jsonify(error_response), 400

        if len(text) > 5000:
            error_response = {
                "success": False,
                "error": "Text too long. Maximum 5000 characters allowed."
            }
            print(f"Error response: {error_response}")
            return jsonify(error_response), 400

        print(f"Analyzing text: {text[:100]}...")

        # Analyze sentiment
        sentiment, confidence, probabilities = analyzer.predict_sentiment(text)
        print(f"Analysis result: sentiment={sentiment}, confidence={confidence}")

        # Construct a structured dictionary for the response
        analysis_result = {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probabilities
        }

        success_response = {
            "success": True,
            "input_text": text,
            "result": analysis_result
        }
        print(f"Success response: {success_response}")

        response = jsonify(success_response)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        error_response = {
            "success": False,
            "error": f"An error occurred: {str(e)}"
        }
        print(f"Exception error response: {error_response}")
        print(f"Exception details: {e}")
        return jsonify(error_response), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train the sentiment analysis model (development only)"""
    try:
        if os.getenv('FLASK_ENV') == 'production':
            return jsonify({
                "error": "Model training is not allowed in production"
            }), 403

        accuracy = analyzer.train_model()

        return jsonify({
            "success": True,
            "message": "Model trained successfully",
            "accuracy": accuracy
        })

    except Exception as e:
        return jsonify({
            "error": f"Training failed: {str(e)}"
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    try:
        # Check if model is loaded
        if analyzer.classifier is None:
            analyzer.load_model()

        if analyzer.classifier is None:
            return jsonify({
                "model_loaded": False,
                "message": "No trained model available"
            })

        # Get model classes
        classes = analyzer.classifier.classes_.tolist() if hasattr(analyzer.classifier, 'classes_') else []

        return jsonify({
            "model_loaded": True,
            "model_type": "Logistic Regression",
            "classes": classes,
            "vectorizer_features": analyzer.tfidf.max_features if analyzer.tfidf else None
        })

    except Exception as e:
        return jsonify({
            "error": f"Error getting model info: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# The following block is for local development only.
# Gunicorn will be used to run the app in production on Render.
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    print(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
