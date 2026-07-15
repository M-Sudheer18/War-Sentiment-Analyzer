import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import pickle

from predictor import predict_sentiment

# Models
@st.cache_resource
def load_best():
    try:
        return hf_hub_download(
            repo_id="Sudheer17/sentiment_lstm",
            filename="best_model.keras"
        )
    except Exception as e:
        st.error("❌ Failed to download Best Model.")
        st.exception(e)
        st.stop()


@st.cache_resource
def load_lstm():
    try:
        return hf_hub_download(
            repo_id="Sudheer17/sentiment_lstm",
            filename="sentiment_lstm.keras"
        )
    except Exception as e:
        st.error("❌ Failed to download Final Model.")
        st.exception(e)
        st.stop()


@st.cache_resource
def load_label():
    try:
        return hf_hub_download(
            repo_id="Sudheer17/sentiment_lstm",
            filename="label_encoder.pkl"
        )
    except Exception as e:
        st.error("❌ Failed to download Label Encoder.")
        st.exception(e)
        st.stop()


@st.cache_resource
def load_tok():
    try:
        return hf_hub_download(
            repo_id="Sudheer17/sentiment_lstm",
            filename="tokenizer.pkl"
        )
    except Exception as e:
        st.error("❌ Failed to download Tokenizer.")
        st.exception(e)
        st.stop()

    

st.set_page_config(
    page_title="War Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load css
with open('style.css') as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# Sidebar
with st.sidebar:

    st.title("🧠 Sentiment Analyzer")

    st.markdown("---")

    model_option = st.radio(
        "Choose LSTM Model",
        [
            "🏆 Best Model",
            "📦 Final Model"
        ]
    )

    if model_option == "🏆 Best Model":
        MODEL_PATH = load_best()
        st.success("Lowest Validation Loss")

    else:
        MODEL_PATH = load_lstm()
        st.info("Model Saved After Last Epoch")

    st.markdown("---")

    st.metric(
        label="Current Model",
        value=model_option
    )

    st.markdown("---")

    st.caption("Built with ❤️ using TensorFlow")
    st.sidebar.markdown("---")

    st.sidebar.caption(
    """
    Built by **Sudheer Muthyala**

    TensorFlow • Streamlit • LSTM • NLTK
    """
    )

# Load Selected Model
@st.cache_resource
def get_model(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error("❌ Failed to load the selected model.")
        st.exception(e)
        st.stop()

model = get_model(MODEL_PATH)


## Download tokenizer and encoder once
TOKENIZER_PATH = load_tok()
LABEL_PATH = load_label()


# Load Tokenizer
@st.cache_resource
def get_tokenizer(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error("❌ Failed to load Tokenizer.")
        st.exception(e)
        st.stop()


# Load Label Encoder
@st.cache_resource
def get_encoder(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error("❌ Failed to load Label Encoder.")
        st.exception(e)
        st.stop()


tokenizer = get_tokenizer(TOKENIZER_PATH)
encoder = get_encoder(LABEL_PATH)

# Center the main content
left, center, right = st.columns([1, 2, 1])

with center:

    st.markdown(
        """
        <div class='title'>
            🧠 Sentiment Analyzer
        </div>

        <div class='subtitle'>
            Ready to analyze your text.
        </div>
        """,
        unsafe_allow_html=True
    )

    user_input = st.text_area(
        "Tweet Input",
        placeholder="Type your Tweet or Sentence here...",
        height=170,
        label_visibility="collapsed"
    )

    analyze = st.button(
        "🚀 Analyze Sentiment",
        use_container_width=False
    )

    if analyze:

        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")

        else:
            with st.spinner("🔄 Analyzing sentiment..."):
                sentiment, confidence, probabilities = predict_sentiment(
                    user_input,
                    model,
                    tokenizer,
                    encoder
                )
            if sentiment is None:
                st.warning("⚠️ Please enter meaningful text.")
            else:
                emoji = {
                    "Positive": "😊",
                    "Neutral": "😐",
                    "Negative": "😞"
                }
                st.success(
                    f"{emoji.get(sentiment, '🧠')} **Sentiment:** {sentiment}"
                )

                st.info(
                    f"🎯 **Confidence:** {confidence:.2%}"
                )

                st.markdown("### 📊 Prediction Probabilities")

                labels = encoder.classes_

                for label, prob in zip(labels, probabilities):
                    st.write(f"**{label}**")
                    st.progress(float(prob))
                    st.caption(f"{prob:.2%}")    