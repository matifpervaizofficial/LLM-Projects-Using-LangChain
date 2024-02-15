## Conversational Q&A Chatbot

from dotenv import load_dotenv
load_dotenv()
import os

import streamlit as st

from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI

## Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("I love Commedy, Let's have a Chat to understand my humour")

chat=ChatOpenAI(temperature=0.5)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content="Yor are a comedian AI assitant")
    ]

## Function to load OpenAI model and get respones
def get_chatmodel_response(question):

    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))

    return answer.content


input=st.text_input("Please type queries: ",key="input")
response=get_chatmodel_response(input)

# Button that triggers the call
submit=st.button("Ask")

## If ask button is clicked
if submit:
    st.subheader("The Response is")
    st.write(response)