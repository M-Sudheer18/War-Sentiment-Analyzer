import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess_text


max_len = 30

def predict_sentiment(text, model, tokenizer, encoder):
    """
    Predict sentiment for input text.
    """

    # Preprocess text
    cleaned_text = preprocess_text(text)

    # Handle empty text after preprocessing
    if cleaned_text == "":
        return None, None, None

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad sequence
    padded = pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    # Predict
    prediction = model.predict(padded, verbose=0)
    probabilities = prediction[0]
    predicted_index = np.argmax(probabilities)

    sentiment = encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])
    
    return sentiment, confidence, probabilities