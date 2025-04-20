import streamlit as st
from PIL import Image
import torch
from model import EncoderCNN, DecoderRNN
import pickle
import torchvision.transforms as transforms

# Define transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Load model components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
encoder = EncoderCNN(embed_size=300).to(device)
decoder = DecoderRNN(embed_size=300, hidden_size=512, vocab_size=8852).to(device)

# Load model weights
encoder_file = r"C:\Users\AYUSH HARINKHEDE\image captioning\models\encoder-3.pkl"
decoder_file = r"C:\Users\AYUSH HARINKHEDE\image captioning\models\decoder-3.pkl"
encoder.load_state_dict(torch.load(encoder_file, map_location=device))
decoder.load_state_dict(torch.load(decoder_file, map_location=device))
encoder.eval()
decoder.eval()

# Load vocabulary
vocab_file = r"C:\Users\AYUSH HARINKHEDE\image captioning\vocab.pkl"
with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)
st.write(f"Vocabulary size: {len(vocab.word2idx)}")

# Caption generation function
def generate_caption(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    features = encoder(image_tensor)
    sampled_ids = decoder.sample(features)  # Returns list
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[int(word_id)]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        sampled_caption.append(word)
    caption = ' '.join(sampled_caption)
    return caption

# Streamlit UI
st.title("🖼️ Automatic Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            caption = generate_caption(image)
        st.markdown(f"**Caption:** {caption}")