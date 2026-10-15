from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

app = Flask(__name__)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Model loaded successfully!")
print("Tokenizer loaded successfully!")


# --------------------------------------------------
# Model Configuration
# --------------------------------------------------

# IMPORTANT:
# Change this value if you used a different
# max_sequence_len while training your model.

MAX_SEQUENCE_LEN = 20


# --------------------------------------------------
# Predict Next Word
# --------------------------------------------------

def predict_next_word(seed_text):

    seed_text = seed_text.lower().strip()

    # Convert text into token sequence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if len(token_list) == 0:
        return None

    # Keep only the last words that fit the model input
    token_list = token_list[-(MAX_SEQUENCE_LEN - 1):]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=MAX_SEQUENCE_LEN - 1,
        padding="pre"
    )

    # Predict
    predictions = model.predict(token_list, verbose=0)

    # Get word with highest probability
    predicted_index = np.argmax(predictions, axis=-1)[0]

    # Convert index back to word
    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

    return None


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.get_json()

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "Please enter some text."
            })

        predicted_word = predict_next_word(text)

        if predicted_word is None:
            return jsonify({
                "success": False,
                "error": "Unable to predict the next word."
            })

        return jsonify({
            "success": True,
            "input": text,
            "prediction": predicted_word
        })

    except Exception as e:

        print("Error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        })


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )