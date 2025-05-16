import streamlit as st
import torch
import torch.nn.functional as F
import whisper
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# Load Whisper ASR model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # Can be 'small', 'medium', etc.

whisper_model = load_whisper_model()

# Load tokenizer and sentiment classifier
@st.cache_resource
def load_sentiment_model():
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=3)
    model.load_state_dict(torch.load("fine_tuned_model_14.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

label_names = ['Negative', 'Neutral', 'Positive']

# UI
st.title("Audio Sentiment Analysis")
st.write("Upload an audio file or enter text manually.")

# AUDIO UPLOAD
audio_file = st.file_uploader("Upload an audio file (.wav, .mp3, etc.)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    
    st.audio("temp_audio.wav")
    st.info("Transcribing audio...")
    result = whisper_model.transcribe("temp_audio.wav")
    transcribed_text = result["text"]
    st.success("Transcription complete.")
    st.markdown(f"**Transcribed Text:** {transcribed_text}")
else:
    transcribed_text = st.text_area("Or enter text manually:", height=150)

# SENTIMENT ANALYSIS
if st.button("Analyze Sentiment"):
    if transcribed_text.strip() == "":
        st.warning("Please provide some text.")
    else:
        # Run sentiment analysis
        encoding = tokenizer(transcribed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = sentiment_model(**encoding)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        st.markdown(f"**Predicted Sentiment:** {label_names[pred]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
