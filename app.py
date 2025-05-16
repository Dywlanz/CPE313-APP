import streamlit as st
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch.nn.functional as F

# Load model and tokenizer
MODEL_PATH = 'fine_tuned_model_14.pth'

# Load tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", 
                                                           num_labels=3
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Sentiment labels
label_names = ['Negative', 'Neutral', 'Positive']

# Streamlit UI
st.title("Sentiment Analysis System by Dylan")
st.write("Enter a sentence and let the model predict its sentiment.")

user_input = st.text_area("Enter text here", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        encoding = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**encoding)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        st.markdown(f"**Predicted Sentiment:** {label_names[pred]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
