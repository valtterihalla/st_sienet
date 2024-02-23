"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
from fastai.vision.all import *
from PIL import Image


nn = load_learner('checkpoint.pkl')
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

st.sidebar.write("## Upload and download :gear:")
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        image = Image.open(my_upload)
        col1.image(image)
        nn.predict(image)[0]
else:
    nn.predict('test.jpg')[0]