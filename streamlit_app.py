import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.title('🕵️Stego-App')

st.info('hide and seek of images')
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
