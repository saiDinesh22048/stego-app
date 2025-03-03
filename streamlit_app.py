import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sqlite3
from PIL import Image
import os
import hashlib
import base64

st.title('🕵️Stego-App')

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
def tensor_to_pil(image_tensor, mean, std):

    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    denormalized_tensor = denormalize(image_tensor.squeeze(0).cpu())
    transform_to_pil = transforms.ToPILImage()
    return transform_to_pil(torch.clamp(denormalized_tensor, 0, 1))

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_users_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def save_stego_image(username, image):
    os.makedirs("stego_images", exist_ok=True)
    filepath = f"stego_images/{username}_stego.png"
    image.save(filepath)
    return filepath

def load_stego_image(username):
    filepath = f"stego_images/{username}_stego.png"
    return Image.open(filepath) if os.path.exists(filepath) else None

st.title("🕵️ Stego-App")
create_users_table()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

def login():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def register():
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registration successful! Please log in.")
        else:
            st.error("Username already exists.")

if not st.session_state["logged_in"]:
    option = st.radio("Choose an option", ["Login", "Register"])
    if option == "Login":
        login()
    else:
        register()
    st.stop()

st.sidebar.header(f"Welcome, {st.session_state['username']}")
