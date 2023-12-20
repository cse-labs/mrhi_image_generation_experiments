import pandas as pd
import generate_event_handler
import json
import logging
import os
import sys
import streamlit as st
from dotenv import load_dotenv
from streamlit.elements.widgets.file_uploader import FileUploaderMixin
from streamlit_option_menu import option_menu
load_dotenv()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger('saicinpainting.training.trainers.base').setLevel(
    logging.WARNING)
logger = logging.getLogger(__name__)
logger.info(
    'Setting PYTORCH_ENABLE_MPS_FALLBACK=1 for mac machines to fallback to CPU')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
with open('style.css') as source:
    st.markdown(f'<style>{source.read()}</style>', unsafe_allow_html=True)
select_option = option_menu(menu_title=None, options=[
                            'On-pack', 'Off-pack'], orientation='horizontal')


def display_on_pack():
    '''"""
This function is used to display the on-pack frame. It allows the user to upload the original on-pack image and the MRHI image of the original image. The user can also input text information about the product. Upon clicking the 'Generate' button, the function will generate an event handler. If the original image is not uploaded and the 'Generate' button is clicked, a warning will be displayed. The function also displays the original and generated MRHI images, as well as the evaluation and validation results if available.
"""'''
    st.header('On-pack frame')
    (col1, col2) = st.columns(2)
    original_image: FileUploaderMixin = col1.file_uploader(
        'Upload the base front pack image for MRHI generation')
    original_mrhi_image = col2.file_uploader(
        'Upload the human generated MRHI image for evaluation')
    text_input = st.text_input(
        'Bottom Lozenges Text', placeholder='Enter information about the product here')
    generate_button = st.button('Generate')
    if original_image is None and generate_button:
        st.warning('Please upload the original image')
    if generate_button:
        (generated_image, validation_results, evaluated_result) = generate_event_handler.on_pack_generate_clicked(
            original_image, original_mrhi_image, text_input)
        (col1, col2) = st.columns(2)
        col1.image(original_image, caption='Original Image',
                   use_column_width=True)
        col2.image(generated_image, caption='Generated MRHI Image',
                   use_column_width=True)
        if evaluated_result is not None and len(evaluated_result) > 0:
            with st.expander('Evaluation Results'):
                df = pd.read_json(json.dumps(evaluated_result), orient='index')
                st.table(df)
        if validation_results is not None and len(validation_results) > 0:
            with st.expander('Validation Results'):
                df = pd.DataFrame(validation_results)
                st.table(df)


def display_off_pack():
    '''"""
This function is used to display the off-pack frame in the user interface. It allows the user to upload the original off-pack image and the MRHI image of the original image. The user can also input the product name and grammage. Upon clicking the 'Generate' button, if the original image is not uploaded, a warning message is displayed. If the original image is uploaded, the function calls the 'off_pack_generate_clicked' function from the 'generate_event_handler' module, passing the uploaded images and the input text as arguments. The original image and the generated MRHI image are then displayed side by side.
"""'''
    st.header('Off-pack frame')
    (col1, col2) = st.columns(2)
    original_image = col1.file_uploader(
        'Upload the base front pack image for MRHI generation')
    (col1, col2) = st.columns(2)
    Snacks_name = col1.text_input(
        'Product name', placeholder='Enter the Product name here')
    Snacks_weight = col2.text_input(
        'Grammage', placeholder='Enter the Product Grammage here')
    generate_button = st.button('Generate')
    if original_image is None and generate_button:
        st.warning('Please upload the original image')
    elif original_image and generate_button:
        (generated_image, evaluation_metrics) = generate_event_handler.off_pack_generate_clicked(
            original_image=original_image, original_mrhi_image=None, left_text=Snacks_name, right_text=Snacks_weight)
        (col1, col2) = st.columns(2)
        col1.image(original_image, caption='Original Image',
                   use_column_width=True)
        col2.image(generated_image, caption='Generated MRHI Image',
                   use_column_width=True)


if select_option == 'On-pack':
    display_on_pack()
elif select_option == 'Off-pack':
    display_off_pack()
