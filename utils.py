import re
import html
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception as e:
    print(f"NLTK download failed: {e}")

# Initialize
# Initialize
lemm = WordNetLemmatizer()

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))


def preprocess_text(txt):
    """
    Preprocess input text before prediction.
    """

    # Convert to string
    txt = str(txt)

    # Lowercase
    txt = txt.lower()

    # Decode HTML entities
    txt = html.unescape(txt)

    # Remove URLs
    txt = re.sub(r"https?://\S+|www\.\S+", "", txt)

    # Remove HTML tags
    txt = re.sub(r"<.*?>", "", txt)

    # Remove punctuation
    txt = txt.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    txt = re.sub(r"\d+", "", txt)

    # Remove extra spaces
    txt = re.sub(r"\s+", " ", txt).strip()

    # Split into words
    words = txt.split()

    # Keep negation words
    negation_words = {"not", "no", "nor", "never"}

    # Remove stopwords except negation words
    words = [
        word for word in words
        if word not in stop_words or word in negation_words
    ]

    # Lemmatization
    words = [lemm.lemmatize(word) for word in words]

    # Join words
    cleaned_text = " ".join(words)

    if not cleaned_text:
        return ""

    return cleaned_text

# if __name__ == "__main__":
#     sample = "Russia attacked Ukraine today!!! https://abc.com"
#     print(preprocess_text(sample))