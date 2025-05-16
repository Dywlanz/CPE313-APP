import streamlit as st
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch.nn.functional as F
import whisper

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # You can change to "small", "medium", etc.

whisper_model = load_whisper_model()

# Load ELECTRA tokenizer and fine-tuned model
@st.cache_resource
def load_model():
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraForSequenceClassification.from_pretrained(
        "google/electra-small-discriminator", num_labels=3
    )
    model.load_state_dict(torch.load("fine_tuned_model_14.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_names = ['Negative', 'Neutral', 'Positive']

# Streamlit UI
st.title("Sentiment Analysis System by Dylan")
st.write("You can enter text or upload an audio file to analyze sentiment.")

# Text input
user_input = st.text_area("Enter text here", height=150)

# Audio input
audio_file = st.file_uploader("Or upload an audio file (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])
transcribed_text = ""

if audio_file is not None:
    st.audio(audio_file)
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    st.write("Transcribing audio...")
    result = whisper_model.transcribe("temp_audio.wav")
    transcribed_text = result["text"]
    st.markdown(f"**Transcribed Text:** {transcribed_text}")

# Combine text from user input or transcription
final_input = user_input if user_input.strip() != "" else transcribed_text

if st.button("Analyze Sentiment"):
    if final_input.strip() == "":
        st.warning("Please enter text or upload an audio file.")
    else:
        # Tokenize and predict
        encoding = tokenizer(final_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**encoding)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        st.markdown(f"**Predicted Sentiment:** {label_names[pred]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
