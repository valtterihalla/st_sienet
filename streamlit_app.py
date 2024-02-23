"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
from fastcore.all import *
from fastai.vision.all import *

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

nn = load_learner('checkpoint.pkl')
nn.predict('test.jpg')[0]
