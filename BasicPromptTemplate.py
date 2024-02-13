"""
    This application will fetch the Celebrity Name from User in the form of Template and
    returns or displays the response from the LLM in the app 
"""

## To Integrate our code with OpenAI API

import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

# For UI Webpages
import streamlit as st

# Seeting the Environment Variables
os.environ['OPENAI_API_KEY'] = openai_key

# streamlit framework
st.title('Celebrity Search using LangChain')
input_text = st.text_input('Mention the name of the Celebrity ')

#  Prompt Template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template= 'Please briefly describe about the Celebrity - {name}'
)

# OpenAI Large Language Models
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm= llm, prompt= first_input_prompt, verbose =True)

if input_text:
    st.write(chain.run(input_text))