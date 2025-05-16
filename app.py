import streamlit as st
import whisper
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch.nn.functional as F

# Load models
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)
model.load_state_dict(torch.load('fine_tuned_model_14.pth', map_location=torch.device('cpu')))
model.eval()

whisper_model = whisper.load_model("base")  # or "small", etc.

label_names = ['Negative', 'Neutral', 'Positive']

st.title("Sentiment Analysis System by Dylan")
st.write("Upload an audio file (e.g., WAV/MP3) or type your sentence.")

# AUDIO input
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

# TEXT input
user_input = st.text_area("Or enter text here", height=150)

if st.button("Analyze"):
    if audio_file:
        # Save uploaded file to disk
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())
        # Transcribe
        transcription = whisper_model.transcribe("temp_audio.wav")["text"]
        user_input = transcription
        st.markdown(f"**Transcribed Text:** {user_input}")

    if user_input.strip() == "":
        st.warning("Please provide either audio or text input.")
    else:
        # Sentiment analysis
        encoding = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**encoding)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        st.markdown(f"**Predicted Sentiment:** {label_names[pred]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
