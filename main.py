import clip
import torch
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Load and preprocess the image
image_path = 'test.png'  # Replace with your image path
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Encode the image
with torch.no_grad():
    image_features = model.encode_image(image) #(1,512)

# The variable `image_features` now contains the encoded features of the image
