"""
    This application will fetch the Celebrity Name from User in the form of Template and
    returns or displays the response from the LLM in the app 
"""

## To Integrate our code with OpenAI API

import os
from credentials import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

# To combine all prompts in Sequential Manner
from langchain.chains import SequentialChain

# To create memory for each Prompt to save the historical Chat
from langchain.memory import ConversationBufferMemory

# For UI Webpages
import streamlit as st

from dotenv import load_dotenv

# Checking if the .env is loaded or not - Returns True
load_dotenv()

# Setting the Environment Variables
os.environ['OPENAI_API_KEY'] = os.getenv('openai_api_key')

# streamlit framework
st.title('Celebrity Search using LangChain')
input_text = st.text_input('Mention the name of the Celebrity')

#  Prompt Template 1
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template= 'Please briefly describe about the Celebrity - {name}'
)

# Creating Memory for first Prompt to save the historical Chat
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')

# OpenAI Large Language Models
llm = OpenAI(temperature=0.8)

# Creating 1st chain to call LLMs
chain = LLMChain(llm= llm, prompt= first_input_prompt, output_key= 'person_info', verbose =True, memory= person_memory)


#  Prompt Template 2
second_input_prompt = PromptTemplate(
    input_variables=['person_info'],
    template= 'When was {person_info} born ?'
)

# Creating Memory for second Prompt to save the historical Chat
date_of_birth_memory = ConversationBufferMemory(input_key='person_info', memory_key='chat_history')

# Creating 2nd chain to call LLMs
chain2 = LLMChain(llm= llm, prompt= second_input_prompt, output_key = 'date_of_birth', verbose =True, memory= date_of_birth_memory)



#  Prompt Template 3
third_input_prompt = PromptTemplate(
    input_variables=['date_of_birth'],
    template= 'Mention 5 major events that happened around {date_of_birth} in the world'
)

# Creating Memory for third Prompt to save the historical Chat
description_memory = ConversationBufferMemory(input_key='date_of_birth', memory_key='description_history')

# Creating 3rd chain to call LLMs
chain3 = LLMChain(llm= llm, prompt= third_input_prompt, output_key = 'description', verbose =True, memory= description_memory)

# Combining all three Prompts in Sequential Manner
sequentialChain = SequentialChain(chains=[chain, chain2, chain3],
                                  input_variables=['name'],
                                  output_variables=['person_info','date_of_birth','description'],
                                  verbose=True)


if input_text:
    st.write(sequentialChain({'name': input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Even Description'):
        st.info(description_memory.buffer)