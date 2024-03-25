# Q&A Chatbot
from langchain_community.llms import OpenAI
import streamlit as st
import os
# from langchain_openai import ChatOpenAI
from main import *


## Function to load OpenAI model and get respones
knowledge_base, df = get_vector_store()


def get_openai_response(question, df):
    result = ask(question, df)
    # llm = OpenAI(temperature=0, streaming=True)
    # response=llm(question)
    response = result["answer"]
    return response


##initialize our streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Healthcare QA chatbot")
input=st.text_input("Input: ", key="input")
response=get_openai_response(input, df)
submit=st.button("Ask the question")

## If ask button is clicked
if submit:
    st.subheader("Retriever Answer w/ chaining:")
    st.write(response)
