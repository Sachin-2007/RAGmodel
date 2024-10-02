from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import langchain_chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

def stop_at_punctuation(text):
    match = re.search(r'(.*?[.!?])\s*$', text)
    if match:
        return match.group(1)
    return text
    
def get_answer(text):
    match = re.search(r'answer:\s*(.*?)(?:\n\s*question:|$)', text, re.DOTALL | re.IGNORECASE)

    if match:
        answer_text = match.group(1).strip()
        print(stop_at_punctuation(answer_text))
        return stop_at_punctuation(answer_text)
    else:
        return "No answer was generated"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_pipeline = pipeline("text-generation", model="gpt2", device = device, temperature = 0.4, truncation=True, max_new_tokens=100)
gpt2_llm = HuggingFacePipeline(pipeline=gpt2_pipeline)

def chunks_from_directory(chunk_size, dir_name):
    loader = DirectoryLoader(dir_name, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = 100)
    chunks = splitter.split_documents(docs)
    return chunks

defaultdb = Chroma(persist_directory='set2db', embedding_function=embedder)
default_retriever = defaultdb.as_retriever(search_kwargs = {'k':2})
default_keyword_retriever = BM25Retriever.from_documents(chunks_from_directory(800, 'set2'))
default_keyword_retriever.k = 2
default_hybrid_retriever = EnsembleRetriever(retrievers = [default_retriever, default_keyword_retriever], weights = [0.5, 0.5])

import streamlit as st
from PyPDF2 import PdfReader

st.title("RAG prompt")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
user_input = st.text_input("Enter your query:")
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Send"):
    st.session_state.chat_history = []
    
    if user_input:
        if pdf_file is None:
            context = " ".join([source.page_content for source in default_hybrid_retriever.invoke(user_input)])
            context = context.replace('\n', ' ')
            prompt = f"{context}\n\nQuestion: {user_input}\nAnswer:"
            response = gpt2_llm(prompt)
            st.session_state.chat_history.append({"user": user_input, "bot": get_answer(response).capitalize()})
   
    if pdf_file is not None:
        pdf_text = PdfReader(pdf_file)
        text = ''
        for page in pdf_text.pages:
            text += page.extract_text()
        with open('db/new.txt', "w") as file:
            file.write(text)
        loader = DirectoryLoader('db', glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap = 100)
        chunks = splitter.split_documents(docs)
        newdb = Chroma.from_documents(chunks, embedder, persist_directory='newdb')
        new_retriever = newdb.as_retriever(search_kwargs = {'k':2})
        new_keyword_retriever = BM25Retriever.from_documents(chunks)
        new_keyword_retriever.k = 2
        new_hybrid_retriever = EnsembleRetriever(retrievers = [new_retriever, new_keyword_retriever], weights = [0.5, 0.5])
        st.write("Context updated")
        
        context = " ".join([source.page_content for source in new_hybrid_retriever.invoke(user_input)])
        prompt = f"{context}\n\nQuestion: {user_input}\nAnswer:"
        response = gpt2_llm(prompt)
        st.session_state.chat_history.append({"user": user_input, "bot": get_answer(response).capitalize()})
        
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.write(f"{chat['bot']}")
