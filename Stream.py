
# import streamlit as st
# import os
# import json
# import base64
# from PIL import Image

# # Set the folder path
# DATA_FOLDER = r"C:\Users\k n ganapati\Desktop\Minerals\output"

# # Function to load image and JSON pairs
# def load_data(folder):
#     image_files = []
#     json_files = {}

#     for file in sorted(os.listdir(folder)):  # Sorted for consistency
#         file_path = os.path.join(folder, file)

#         if file.lower().endswith(("png", "jpg", "jpeg", "gif")):
#             image_files.append(file_path)  # Store image path
        
#         elif file.lower().endswith(".json"):
#             with open(file_path, "r", encoding="utf-8") as f:
#                 try:
#                     json_data = json.load(f)
#                     json_files[file] = json_data  # Store JSON content
#                 except json.JSONDecodeError:
#                     st.error(f"Error reading {file}")

#     return image_files, json_files

# # Function to encode image to base64
# def encode_image(image_path):
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# # Load images and JSON metadata
# image_files, json_files = load_data(DATA_FOLDER)

# # Streamlit UI
# st.set_page_config(layout="wide")  # Full-screen width
# st.title("🪨 Minerals Image Viewer")

# # Display images in a scrollable format
# for image_path in image_files:
#     image_name = os.path.basename(image_path)
#     image_base64 = encode_image(image_path)

#     st.markdown(
#         f"""
#         <div style="
#             border: 4px solid #3498db; 
#             padding: 10px; 
#             border-radius: 10px; 
#             display: flex; 
#             justify-content: center; 
#             align-items: center;
#             width: 100%;
#             height: 75vh;  /* ✅ Ensures image fits within window */
#             overflow: hidden;
#         ">
#             <img src="data:image/png;base64,{image_base64}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # Display JSON metadata if available
#     json_name = os.path.splitext(image_name)[0] + ".json"
#     if json_name in json_files:
#         st.json(json_files[json_name])  # Display JSON metadata

#     st.markdown("---")  # Add spacing between images
import streamlit as st
import os
import base64
from PIL import Image

# Set the folder path
DATA_FOLDER = r"output"

# Function to load images

def load_images(folder):
    image_files = []
    for file in sorted(os.listdir(folder)):  # Sorted for consistency
        file_path = os.path.join(folder, file)
        if file.lower().endswith(("png", "jpg", "jpeg", "gif")):
            image_files.append(file_path)  # Store image path
    return image_files

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load images
image_files = load_images(DATA_FOLDER)

# Streamlit UI
st.set_page_config(layout="wide")  # Full-screen width
st.title("🪨 Minerals Image Viewer")

# Display images in a scrollable format
for image_path in image_files:
    image_base64 = encode_image(image_path)
    st.markdown(
        f"""
        <div style="
            border: 4px solid #3498db;
            padding: 10px;
            border-radius: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 75vh;  /* ✅ Ensures image fits within window */
            overflow: hidden;
        ">
            <img src="data:image/png;base64,{image_base64}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")  # Add spacing between images

