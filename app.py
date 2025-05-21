import streamlit as st
import torch
import torch.nn.functional as F
import whisper
import re
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# Profanity censor function
def censor_profanity(text):
    bad_words = ['damn', 'shit', 'fuck', 'bitch', 'asshole']  # Add more
    for word in bad_words:
        pattern = re.compile(rf'\b{re.escape(word)}\b', flags=re.IGNORECASE)
        text = pattern.sub(lambda m: '*' * len(m.group()), text)
    return text

# Load Whisper ASR model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Load tokenizer and sentiment model
@st.cache_resource
def load_sentiment_model():
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=3)
    model.load_state_dict(torch.load("fine_tuned_model_14.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

label_names = ['Negative', 'Neutral', 'Positive']
label_colors = ['#FF4B4B', '#FFD700', '#4CAF50']

# Page design
st.set_page_config(page_title="Audio Sentiment Analysis", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎧 Audio Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an audio file or enter text manually to detect emotional tone.</p>", unsafe_allow_html=True)

# File uploader
st.markdown("### 🔊 Upload an audio file")
audio_file = st.file_uploader("Supported formats: .wav, .mp3, .m4a", type=["wav", "mp3", "m4a"])

transcribed_text = ""

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())

    st.audio("temp_audio.wav")
    st.info("🕒 Transcribing audio...")
    result = whisper_model.transcribe("temp_audio.wav")
    raw_text = result["text"]
    transcribed_text = censor_profanity(raw_text)
    st.success("✅ Transcription complete!")
    st.markdown("### 📝 Transcribed Text (Censored):")
    st.code(transcribed_text, language='markdown')
else:
    st.markdown("### 📝 Or enter text manually:")
    transcribed_text = st.text_area("Paste or type your message here", height=150)

# Sentiment analysis
st.markdown("### 💬 Sentiment Prediction")

if st.button("🔍 Analyze Sentiment"):
    if transcribed_text.strip() == "":
        st.warning("⚠️ Please provide some text.")
    else:
        encoding = tokenizer(transcribed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = sentiment_model(**encoding)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        sentiment_color = label_colors[pred]
        sentiment_label = label_names[pred]

        st.markdown(
            f"<div style='background-color: {sentiment_color}; padding: 10px; border-radius: 10px;'>"
            f"<h3 style='color: white;'>Predicted Sentiment: {sentiment_label}</h3>"
            f"<p style='color: white;'>Confidence: {confidence:.2%}</p>"
            "</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Whisper and ELECTRA</p>", unsafe_allow_html=True)
