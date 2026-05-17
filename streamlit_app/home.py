import streamlit as st

import warnings
warnings.filterwarnings("ignore")

from transformers.utils import logging
logging.set_verbosity_error()

st.set_page_config(
    page_title='AI Ecommerce Customer Platform',
    layout='wide'
)

st.title('AI Ecommerce Customer Intelligence Platform')

st.write('Welcome to the platform')

st.write('Modules Available:')

st.write('- Churn Prediction')
st.write('- CLV Prediction')
st.write('- Recommendations')
st.write('- NLP Analysis')