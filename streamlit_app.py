import streamlit as st
import sqlite3
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import base64

# Initialize database
def init_db():
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

# User authentication functions
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Dummy steganography model (Replace with actual model)
class SimpleStegoModel(nn.Module):
    def forward(self, cover, secret):
        return cover  # Dummy model (Replace with actual implementation)

def encode_image(cover_image, secret_image):
    model = SimpleStegoModel()
    cover_tensor = torch.tensor(np.array(cover_image)).float()
    secret_tensor = torch.tensor(np.array(secret_image)).float()
    stego_image = model(cover_tensor, secret_tensor)
    stego_pil = Image.fromarray(stego_image.numpy().astype(np.uint8))
    return stego_pil

def decode_image(stego_image):
    return stego_image  # Dummy decoder (Replace with actual implementation)

# Streamlit UI
st.title("Secure Image Steganography")
init_db()

# Authentication
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create an Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(new_user, new_pass):
            st.success("Registered successfully! Please login.")
        else:
            st.warning("Username already taken.")

elif choice == "Login":
    st.subheader("Login to your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
        else:
            st.error("Invalid credentials")

if "logged_in" in st.session_state and st.session_state["logged_in"]:
    st.sidebar.subheader(f"Welcome, {st.session_state['username']}")
    option = st.sidebar.radio("Choose Action", ["Encode Image", "Decode Image"])

    if option == "Encode Image":
        st.subheader("Upload Cover and Secret Image")
        cover = st.file_uploader("Choose Cover Image", type=["png", "jpg", "jpeg"])
        secret = st.file_uploader("Choose Secret Image", type=["png", "jpg", "jpeg"])
        if cover and secret:
            cover_image = Image.open(cover).convert("RGB")
            secret_image = Image.open(secret).convert("RGB")
            stego_image = encode_image(cover_image, secret_image)
            st.image(stego_image, caption="Stego Image", use_container_width=True)
            buf = io.BytesIO()
            stego_image.save(buf, format="PNG")
            st.download_button("Download Stego Image", buf.getvalue(), "stego.png", "image/png")
    
    elif option == "Decode Image":
        st.subheader("Upload Stego Image to Reveal Secret Image")
        stego = st.file_uploader("Choose Stego Image", type=["png", "jpg", "jpeg"])
        if stego:
            stego_image = Image.open(stego).convert("RGB")
            secret_image = decode_image(stego_image)
            st.image(secret_image, caption="Revealed Secret Image", use_container_width=True)
