"""
    This file get the basic question from the user. 
    Uses LLM Models and then displayes the response of the Model in the app
"""

## To Integrate our code with OpenAI API
import os
from credentials import openai_key
from langchain_openai import OpenAI

# For UI Webpages
import streamlit as st

from dotenv import load_dotenv

# Checking if the .env is loaded or not - Returns True
load_dotenv()

# Setting the Environment Variables
os.environ['OPENAI_API_KEY'] = os.getenv('openai_api_key')

# streamlit framework
st.title('Search Celebrity using LangChain with OpenAI API')
input_text = st.text_input('Mention the name of the ')

# OpenAI Large Language Models
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
