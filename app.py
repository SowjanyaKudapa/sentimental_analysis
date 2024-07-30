from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

# Paths to your model and vectorizer
model_path = r"C:\Users\divya mullapudi\OneDrive\Attachments\Desktop\FSD\models\model_dt.pkl"
vectorizer_path = r"C:\Users\divya mullapudi\OneDrive\Attachments\Desktop\FSD\models\countVectorizer.pkl"

# Load pre-trained model and vectorizer
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

def preprocess(text):
    """Preprocess the input text."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase to ensure consistency
    print(f"Preprocessed text: {text}")  # Debugging line
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the sentiment of the input text."""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    text = preprocess(text)
    text_features = vectorizer.transform([text])

    print(f"Text features: {text_features.toarray()}")  # Debugging line

    try:
        prediction = model.predict(text_features)
        sentiment = 'positive' if prediction[0] == 1 else 'negative'
        print(f"Prediction: {sentiment}")  # Debugging line
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
