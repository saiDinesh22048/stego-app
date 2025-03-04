import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.title('🕵Stego-App')

st.info('hide and seek of images')

# Load the trained models
class PreparationNetwork(nn.Module):
    def __init__(self):
        super(PreparationNetwork, self).__init__()
        self.branch1_conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.branch1_conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.branch2_conv1 = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.branch2_conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.branch3_conv1 = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        self.branch3_conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.relu(self.branch1_conv1(x))
        b1 = self.relu(self.branch1_conv2(b1))
        b2 = self.relu(self.branch2_conv1(x))
        b2 = self.relu(self.branch2_conv2(b2))
        b3 = self.relu(self.branch3_conv1(x))
        b3 = self.relu(self.branch3_conv2(b3))
        return torch.cat((b1, b2, b3), dim=1)  # 65 channels

class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.input_conv = nn.Conv2d(68, 50, kernel_size=3, padding=1)
        self.branch1_convs = nn.ModuleList([nn.Conv2d(50, 50, kernel_size=3, padding=1) for _ in range(5)])
        self.branch2_convs = nn.ModuleList([nn.Conv2d(50, 50, kernel_size=3, padding=1) for _ in range(2)])
        self.branch3_convs = nn.ModuleList([nn.Conv2d(50, 50, kernel_size=3, padding=1) for _ in range(2)])
        self.final_conv = nn.Conv2d(150, 3, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()

    def forward(self, cover, secret):
        x = torch.cat((cover, secret), dim=1)
        x = self.relu(self.input_conv(x))
        b1, b2, b3 = x, x, x
        for conv_layer in self.branch1_convs:
            b1 = self.relu(conv_layer(b1))
        for conv_layer in self.branch2_convs:
            b2 = self.relu(conv_layer(b2))
        for conv_layer in self.branch3_convs:
            b3 = self.relu(conv_layer(b3))
        combined = torch.cat((b1, b2, b3), dim=1)
        return self.final_conv(combined)

class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.branch1_convs = nn.ModuleList([nn.Conv2d(50, 50, kernel_size=3, padding=1) for _ in range(5)])
        self.branch2_convs = nn.ModuleList([nn.Conv2d(50, 50, kernel_size=3, padding=1) for _ in range(2)])
        self.branch3_convs = nn.ModuleList([nn.Conv2d(50, 50, kernel_size=3, padding=1) for _ in range(2)])
        self.final_conv = nn.Conv2d(150, 3, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.initial_conv(x))
        b1, b2, b3 = x, x, x
        for conv_layer in self.branch1_convs:
            b1 = self.relu(conv_layer(b1))
        for conv_layer in self.branch2_convs:
            b2 = self.relu(conv_layer(b2))
        for conv_layer in self.branch3_convs:
            b3 = self.relu(conv_layer(b3))
        combined = torch.cat((b1, b2, b3), dim=1)
        return self.final_conv(combined)
# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prep_net = PreparationNetwork().to(device)
hide_net = HidingNetwork().to(device)
reveal_net = RevealNetwork().to(device)

# Load the saved weights
prep_net.load_state_dict(torch.load("preparation_network.pth", map_location=device))
hide_net.load_state_dict(torch.load("hiding_network.pth", map_location=device))
reveal_net.load_state_dict(torch.load("reveal_network.pth", map_location=device))

# Set to evaluation mode
prep_net.eval()
hide_net.eval()
reveal_net.eval()

st.write("Models loaded successfully!")

# Convert tensors to images
def tensor_to_pil(image_tensor, mean, std):

    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    denormalized_tensor = denormalize(image_tensor.squeeze(0).cpu())
    transform_to_pil = transforms.ToPILImage()
    return transform_to_pil(torch.clamp(denormalized_tensor, 0, 1))
st.sidebar.header("Upload Images")
cover_file = st.sidebar.file_uploader("Upload Cover Image", type=["jpg", "png", "jpeg"])
secret_file = st.sidebar.file_uploader("Upload Secret Image", type=["jpg", "png", "jpeg"])

if cover_file and secret_file:
    cover_image = Image.open(cover_file).convert("RGB")
    secret_image = Image.open(secret_file).convert("RGB")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)
    secret_tensor = transform(secret_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prepared_secret = prep_net(secret_tensor)
        stego_image = hide_net(cover_tensor, prepared_secret)
        stego_pil = tensor_to_pil(stego_image, mean, std)
        stego_tensor = transform(stego_pil).unsqueeze(0).to(device)
        revealed_secret = reveal_net(stego_tensor)



    cover_pil  = tensor_to_pil(cover_tensor, mean, std)
    secret_pil = tensor_to_pil(secret_tensor, mean, std)
    stego_pil = tensor_to_pil(stego_image, mean, std)
    revealed_pil = tensor_to_pil(revealed_secret, mean, std)

    st.subheader("Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.image(cover_pil, caption="Cover Image", use_container_width=True)
    col2.image(secret_pil, caption="Secret Image",use_container_width=True)
    col3.image(stego_pil, caption="Stego Image", use_container_width=True)
    col4.image(revealed_pil, caption="Revealed Secret", use_container_width=True)

    # Download Stego Image
    st.subheader("Download Stego Image")
    st.download_button("Download", stego_pil.tobytes(),"stego_image.png","image/png")
