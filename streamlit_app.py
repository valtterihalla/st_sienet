"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
from fastai.vision.all import *
from PIL import Image


nn = load_learner('checkpoint.pkl')
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

st.write("## Lataa kuva sienestä :gear:")
my_upload = st.file_uploader("", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("Kuvatiedosto on liian suuri, yli 5 megatavua.")
    else:
        image = Image.open(my_upload)
        st.image(image)
        st.write("##", nn.predict(image)[0], "##")
else:
    image = Image.open("./test_image.jpg")
    st.image(image)
    st.write("##", nn.predict(image)[0], "##")
