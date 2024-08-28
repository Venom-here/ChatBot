## Since we are using Ollama Embedding, it takes a lot of time to embed the entire pdfs and hence requires a lot of time to run on streamlit

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

## 
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)


def create_vector_embedding():
    if "vectors" not in st.session_state: ## session_state helps to remember vectorStore DB
        st.session_state.embeddings=(OllamaEmbeddings(model="gemma2:2b"))
        st.session_state.loader=CSVLoader('query.csv') ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.vectors=FAISS.from_documents(st.session_state.docs,st.session_state.embeddings)
st.title("GTEC ChatBot")

user_prompt = st.text_input("Enter your query")


if st.button("Answer"):
    create_vector_embedding()
    st.write("Good to Go")


import time


if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)


    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt}) # {input} --> prompt
    print(f"Response time :{time.process_time()-start}")


    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')