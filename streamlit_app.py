import streamlit as st
import sqlite3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from io import BytesIO
import uuid

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

def pil_to_bytes(pil_image, format='PNG'):
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()  # Get the byte value of the image
    return img_byte_arr
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
    
def delete_image(msg_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE id=?", (msg_id,))
    c.execute("DELETE FROM SQLITE_SEQUENCE WHERE name='messages'")
    conn.commit()
    conn.close()
    
def remove_user(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username,))
    c.execute("DELETE FROM messages WHERE sender=? OR receiver=?", (username, username))
    conn.commit()
    conn.close()

def get_all_messages():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id, sender, receiver FROM messages")
    messages = c.fetchall()
    conn.close()
    return messages

# Streamlit UI
st.title("🔒 Stego Image Sharing App")
st.sidebar.title("Authentication")
menu = ["Login", "Register", "Admin Login"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == "Register":
    st.session_state.pop("logged_in", None)
    st.session_state.pop("admin_logged_in", None)
    st.subheader("User Registration")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(new_user, new_pass):
            st.success("Registered Successfully! You can now login.")
        else:
            st.error("Username already exists!")

elif choice == "Login":
    
    st.session_state.pop("admin_logged_in", None)
    st.subheader("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"Welcome {username}")
        else:
            st.error("Invalid Credentials!")

elif choice == "Admin Login":
    st.session_state.pop("logged_in", None)
   
    st.subheader("Admin Login")
    admin_user = st.text_input("Admin Username")
    admin_pass = st.text_input("Admin Password", type="password")
    if st.button("Login"):
        if admin_user == "admin" and admin_pass == "admin":
            st.session_state["admin_logged_in"] = True
            st.success("Admin Logged In Successfully!")
        else:
            st.error("Invalid Admin Credentials!")


# After login
if "logged_in" in st.session_state:
    st.subheader(f"Welcome, {st.session_state['username']}")

    tab1, tab2 = st.tabs(["Send Stego Image", "Received Stego Images"])

    # Sending Stego Images
    with tab1:
        users = get_users()
        users.remove(st.session_state["username"])
        receiver = st.selectbox("Send To", users)
        # Create columns for displaying images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼 Cover Image")
            cover_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
            if cover_file:
                cover = Image.open(cover_file)
                st.image(cover, caption="Cover Image", use_container_width=True)
        
        with col2:
            st.subheader("🔒 Secret Image")
            secret_file = st.file_uploader("Upload Secret Image", type=["png", "jpg", "jpeg"])
            if secret_file:
                secret = Image.open(secret_file)
                st.image(secret, caption="Secret Image",use_container_width=True)
        
        if st.button("Send Stego Image"):
            if cover_file and secret_file and receiver:
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

                image_bytes = pil_to_bytes(stego_pil, format='JPEG')
                st.image(stego_pil, caption="Stego Image",width=200)
                send_stego_image(st.session_state["username"], receiver, image_bytes)
                st.success("Stego image sent!")
       
        if st.button(f"Logout", key=f"logout_1"):
            st.session_state.clear()
            st.rerun()

    # Viewing Received Stego Images
    with tab2:
        received_images = get_received_images(st.session_state["username"])
        for msg_id, sender, img_blob in received_images:
            st.subheader(f"From: {sender}")
            image = Image.open(io.BytesIO(img_blob))
            st.image(image, caption="Received Stego Image", width=200)
            
            if st.button(f"Extract Secret (ID: {msg_id})"):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            
                # Image transformation
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
                with torch.no_grad():
                    
                    
                    stego_tensor = transform(image).unsqueeze(0).to(device)
                    revealed_secret = reveal_net(stego_tensor)
                    revealed_pil = tensor_to_pil(revealed_secret, mean, std)

                
                st.image(revealed_pil, caption="Extracted Secret Image", use_container_width=True)
                st.download_button("Download", stego_pil.tobytes(), "stego_image.png", "image/png")
            if st.button(f"Delete Image {msg_id}"):
                delete_image(msg_id)
                st.success("Image Deleted Successfully!")
        
        if st.button(f"Logout", key=f"logout_2"):
            st.session_state.clear()
            st.rerun()
if "admin_logged_in" in st.session_state:
    st.subheader("Admin Dashboard")
    users = get_users()
    st.write("Registered Users:")
    for user in users:
        if user != "admin":
            col1, col2 = st.columns(2)
            with col1:
                st.write(user)
            with col2:
                if st.button(f"Remove {user}"):
                    remove_user(user)
                    st.success(f"User {user} removed!")
    
    st.subheader("All Messages")
    messages = get_all_messages()
    for msg_id, sender, recipient in messages:
        st.write(f"ID: {msg_id}, Sender: {sender}, Receiver: {recipient}")
        if st.button(f"Remove Image {msg_id}"):
            delete_image(msg_id)
            st.success(f"Image {msg_id} removed!")
    
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
