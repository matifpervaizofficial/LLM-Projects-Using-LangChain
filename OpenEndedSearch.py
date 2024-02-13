"""
    This file get the basic question from the user. 
    Uses LLM Models and then displayes the response of the Model in the app
"""

## To Integrate our code with OpenAI API
import os
from constants import openai_key
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

# For UI Webpages
import streamlit as st


# Seeting the Environment Variables
os.environ['OPENAI_API_KEY'] = openai_key

# streamlit framework
st.title('Search Celebrity using LangChain with OpenAI API')
input_text = st.text_input('Mention the name of the ')

# OpenAI Large Language Models
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
