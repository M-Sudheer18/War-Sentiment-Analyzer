# 🧠 War Sentiment Analyzer using LSTM
### Deep Learning-Based Twitter Sentiment Classification with TensorFlow, Streamlit & Hugging Face

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model_Hosting-yellow?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# 🌟 Overview

**War Sentiment Analyzer** is an end-to-end **Natural Language Processing (NLP)** application powered by a **Long Short-Term Memory (LSTM)** deep learning model. The application predicts whether a user-provided sentence expresses a **Positive**, **Neutral**, or **Negative** sentiment while providing prediction confidence and probability distributions.

Unlike traditional machine learning approaches, this project leverages sequential deep learning to understand contextual relationships between words, making sentiment prediction significantly more accurate for natural language.

The application features a professional **Streamlit** interface, automatic model downloading from **Hugging Face Hub**, intelligent preprocessing, multiple model selection, and real-time sentiment prediction.

---

# 🚀 Live Demo

> 🔗 **Coming Soon**

---

# ✨ Key Features

## 🤖 AI & Deep Learning

- Multi-class Sentiment Classification
- Deep LSTM Neural Network
- TensorFlow/Keras Implementation
- Softmax Probability Prediction
- Confidence Score Estimation
- Supports Positive, Neutral & Negative Classes

---

## 🌐 Interactive Web Application

- Modern Streamlit Interface
- Responsive Layout
- Custom CSS Styling
- Sidebar Navigation
- Live Text Prediction
- Loading Animation
- Professional Result Cards
- Interactive Probability Bars

---

## ⚡ Smart Model Management

- Automatic Model Download from Hugging Face
- Cached Resource Loading
- Multiple Model Selection

  - 🏆 Best Model
  - 📦 Final Model

- Lightweight GitHub Repository
- Fast Application Startup

---

## 🧹 Intelligent NLP Pipeline

- Lowercase Conversion
- HTML Entity Decoding
- URL Removal
- HTML Tag Removal
- Punctuation Removal
- Number Removal
- Stopword Removal
- Negation Preservation
- Lemmatization
- Tokenization
- Sequence Padding

---

# 📸 Application Workflow

```
                User Input
                     │
                     ▼
        Text Preprocessing Pipeline
                     │
                     ▼
             Tokenizer Encoding
                     │
                     ▼
           Sequence Padding (30)
                     │
                     ▼
             TensorFlow LSTM Model
                     │
                     ▼
            Softmax Probability Layer
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Positive      Neutral      Negative
        │
        ▼
 Confidence Score + Probability Chart
```

---

# 🏗️ System Architecture

```
                        ┌───────────────────────┐
                        │    User Interface     │
                        │      Streamlit        │
                        └──────────┬────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │      User Input Text      │
                    └──────────┬───────────────┘
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │    Text Preprocessing        │
                 │ • Lowercase                 │
                 │ • URL Removal               │
                 │ • HTML Cleaning             │
                 │ • Lemmatization             │
                 │ • Stopword Removal          │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │      Tokenizer               │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │     Sequence Padding         │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │ TensorFlow LSTM Model        │
                 └──────────┬──────────────────┘
                            │
                            ▼
                 ┌─────────────────────────────┐
                 │  Probability Prediction      │
                 └──────────┬──────────────────┘
                            │
                            ▼
               Positive | Neutral | Negative
```

---

# 🧠 Deep Learning Architecture

The prediction engine is developed using **TensorFlow/Keras Sequential API**.

```
Input Layer
      │
      ▼
Embedding Layer
(Vocabulary Embeddings)
      │
      ▼
LSTM (128 Units)
      │
Dropout (0.2)
      │
      ▼
LSTM (64 Units)
      │
Dropout (0.2)
      │
      ▼
LSTM (32 Units)
      │
Dropout (0.2)
      │
      ▼
GlobalMaxPooling1D
      │
      ▼
Dense Layer (64 ReLU)
      │
Dropout (0.2)
      │
      ▼
Output Layer
Softmax (3 Classes)
      │
      ▼
Positive
Neutral
Negative
```

---

# 🔄 Prediction Pipeline

```
Raw User Text
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Padding
      │
      ▼
LSTM Neural Network
      │
      ▼
Softmax Prediction
      │
      ▼
Label Decoder
      │
      ▼
Confidence Score
      │
      ▼
Probability Distribution
```

---

# 🧹 Text Preprocessing Pipeline

Every input undergoes exactly the same preprocessing used during model training.

| Step | Description |
|-------|-------------|
| Lowercase | Converts all text to lowercase |
| HTML Decode | Removes HTML entities |
| URL Removal | Removes web links |
| HTML Removal | Removes HTML tags |
| Punctuation Removal | Removes punctuation |
| Number Removal | Removes numeric values |
| Extra Space Removal | Removes unnecessary spaces |
| Tokenization | Splits text into words |
| Stopword Removal | Removes common words |
| Negation Preservation | Keeps words like **not**, **never** |
| Lemmatization | Converts words into root forms |

---

### Example

#### Input

```
Russia attacked Ukraine today!!!

https://example.com
```

↓

#### Processed Text

```
russia attacked ukraine today
```

---

# 📂 Project Structure

```
War-Sentiment-Analyzer/
│
├── app.py
├── predictor.py
├── utils.py
├── style.css
├── requirements.txt
├── .gitignore
├── README.md
│
├── Data/
│   └── sentiment.csv
│
├── Models/
│   ├── best_model.keras
│   ├── sentiment_lstm.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── assets/
│
└── sent_env/
```

---

# 💻 Technology Stack

| Layer | Technologies |
|---------|-------------|
| Frontend | Streamlit, HTML, CSS |
| Backend | Python |
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK |
| Machine Learning | Scikit-Learn |
| Numerical Computing | NumPy |
| Model Hosting | Hugging Face Hub |

---

# 🤖 Machine Learning Stack

- TensorFlow 2.x
- Keras Sequential API
- Embedding Layer
- Multi-layer LSTM
- Dropout Regularization
- Global Max Pooling
- Dense Neural Networks
- Softmax Activation
- Adam Optimizer
- Sparse Categorical Crossentropy
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

---

# 📊 Model Training

## Dataset

Custom Twitter Sentiment Dataset

---

## Data Preparation

- English Tweets Filtered
- Duplicate Removal
- Data Cleaning
- Tokenization
- Sequence Padding
- Label Encoding

---

## Hyperparameters

| Parameter | Value |
|------------|---------|
| Vocabulary Size | 10,000 |
| Embedding Dimension | 128 |
| Sequence Length | 30 |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

---

# ☁️ Model Deployment

Large model files are hosted on **Hugging Face Hub** instead of GitHub.

Application startup flow:

```
Launch Streamlit
        │
        ▼
Download Models
        │
        ▼
Cache Resources
        │
        ▼
Load TensorFlow Models
        │
        ▼
Load Tokenizer
        │
        ▼
Load Label Encoder
        │
        ▼
Application Ready
```

This approach keeps the GitHub repository lightweight while ensuring fast model loading.

---

# 📊 Prediction Output

The application predicts:

- 😊 Sentiment
- 🎯 Confidence Score
- 📈 Probability Distribution

### Example

```
Input

I absolutely love this phone.
```

↓

```
Prediction

Positive 😊

Confidence

98.42%

Probability

Positive : 98%

Neutral : 1%

Negative : 1%
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/War-Sentiment-Analyzer.git
```

---

## Navigate

```bash
cd War-Sentiment-Analyzer
```

---

## Create Virtual Environment

```bash
python -m venv sent_env
```

---

## Activate Environment

### Windows

```bash
sent_env\Scripts\activate
```

### Linux / macOS

```bash
source sent_env/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```
streamlit
tensorflow
numpy
scikit-learn
nltk
huggingface_hub
h5py
```

---

# 📈 Future Improvements

- CSV Batch Prediction
- Export Predictions
- User Prediction History
- REST API (FastAPI)
- Docker Deployment
- CI/CD Integration
- Model Quantization
- Attention-Based LSTM
- GRU Models
- BERT
- RoBERTa
- DistilBERT
- Real-Time Twitter/X Streaming
- Explainable AI (LIME/SHAP)

---

# 👨‍💻 Developer

## Sudheer Muthyala

**Final Year B.Tech (Electronics & Communication Engineering)**

Passionate about:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Data Science
- MLOps

---

# 🙏 Acknowledgements

Special thanks to the open-source community and the developers behind:

- TensorFlow
- Keras
- Streamlit
- Hugging Face
- NLTK
- Scikit-Learn

for providing exceptional tools that made this project possible.

---

# ⭐ Support

If you found this project helpful,

### ⭐ Star this repository

### 🍴 Fork the project

### 🛠️ Contribute improvements

### 📢 Share it with others

---

<p align="center">

### 🚀 Built with ❤️ using TensorFlow, LSTM, Streamlit & Hugging Face

**Made by Sudheer Muthyala**

</p>
