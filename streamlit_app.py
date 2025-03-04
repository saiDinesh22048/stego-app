import streamlit as st
import sqlite3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT,
                    receiver TEXT,
                    stego_image BLOB)''')
    conn.commit()
    conn.close()

init_db()

# Authentication functions
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

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def get_users():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users")
    users = [user[0] for user in c.fetchall()]
    conn.close()
    return users

def send_stego_image(sender, receiver, image_bytes):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO messages (sender, receiver, stego_image) VALUES (?, ?, ?)", (sender, receiver, image_bytes))
    conn.commit()
    conn.close()

def get_received_images(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id, sender, stego_image FROM messages WHERE receiver=?", (username,))
    messages = c.fetchall()
    conn.close()
    return messages

# Streamlit UI
st.title("🔒 Stego Image Sharing App")
st.sidebar.title("Authentication")

# Login/Register
choice = st.sidebar.radio("Select an option", ["Login", "Register"])
if choice == "Register":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Register"):
        if register_user(username, password):
            st.sidebar.success("Registered successfully. Please login.")
        else:
            st.sidebar.error("Username already exists!")
elif choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid credentials")

# After login
if "logged_in" in st.session_state:
    st.subheader(f"Welcome, {st.session_state['username']}")

    tab1, tab2 = st.tabs(["Send Stego Image", "Received Stego Images"])

    # Sending Stego Images
    with tab1:
        users = get_users()
        users.remove(st.session_state["username"])
        receiver = st.selectbox("Send To", users)
        cover_file = st.file_uploader("Upload Cover Image", type=["jpg", "png", "jpeg"])
        secret_file = st.file_uploader("Upload Secret Image", type=["jpg", "png", "jpeg"])
        if st.button("Send Stego Image"):
            if cover_file and secret_file and receiver:
                # For simplicity, storing cover image as stego (replace with your model output)
                image_bytes = cover_file.read()
                send_stego_image(st.session_state["username"], receiver, image_bytes)
                st.success("Stego image sent!")

    # Viewing Received Stego Images
    with tab2:
        received_images = get_received_images(st.session_state["username"])
        for msg_id, sender, img_blob in received_images:
            st.subheader(f"From: {sender}")
            image = Image.open(io.BytesIO(img_blob))
            st.image(image, caption="Received Stego Image",use_container_width=True)
            
            # Extract Secret Image (Placeholder for actual model)
            if st.button(f"Extract Secret (ID: {msg_id})"):
                extracted_secret = image  # Replace with model output
                buf = io.BytesIO()
                extracted_secret.save(buf, format="PNG")
                st.image(extracted_secret, caption="Extracted Secret Image", use_container_width=True)
                st.download_button("Download Secret Image", buf.getvalue(), f"secret_image_{msg_id}.png", "image/png")
