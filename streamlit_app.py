import streamlit as st
import sqlite3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from io import BytesIO
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
    users = c.fetchall()
    conn.close()
    return [user[0] for user in users]

def send_stego_image(sender, receiver, stego_image):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    stego_bytes = BytesIO()
    stego_image.save(stego_bytes, format="PNG")
    c.execute("INSERT INTO messages (sender, receiver, stego_image) VALUES (?, ?, ?)", 
              (sender, receiver, stego_bytes.getvalue()))
    conn.commit()
    conn.close()

def get_received_images(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT sender, stego_image FROM messages WHERE receiver=?", (username,))
    images = c.fetchall()
    conn.close()
    return images

st.title("🕵 Stego-App: Secure Image Sharing")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    if st.button("Register"):
        if register_user(new_user, new_password):
            st.success("Account created successfully! Proceed to login.")
        else:
            st.error("Username already exists. Try another one.")

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid credentials")

if 'logged_in' in st.session_state and st.session_state['logged_in']:
    st.subheader(f"Welcome, {st.session_state['username']}!")
    option = st.selectbox("Select Option", ["Send Stego Image", "View Received Stego Images"])
    users = get_users()
    users.remove(st.session_state['username'])  # Remove self from list

    if option == "Send Stego Image":
        receiver = st.selectbox("Select a User to Send Stego Image", users)
        cover_file = st.file_uploader("Upload Cover Image", type=["jpg", "png", "jpeg"])
        secret_file = st.file_uploader("Upload Secret Image", type=["jpg", "png", "jpeg"])
        if st.button("Create & Send Stego Image"):
            if cover_file and secret_file:
                cover_image = Image.open(cover_file).convert("RGB")
                secret_image = Image.open(secret_file).convert("RGB")
                # Dummy stego processing - Replace with your model
                stego_image = cover_image  # Placeholder
                send_stego_image(st.session_state['username'], receiver, stego_image)
                st.success("Stego Image Sent Successfully!")
    
    elif option == "View Received Stego Images":
        images = get_received_images(st.session_state['username'])
        if images:
            for sender, img_data in images:
                st.write(f"From: {sender}")
                img = Image.open(BytesIO(img_data))
                st.image(img, caption="Received Stego Image")
        else:
            st.info("No images received yet.")
