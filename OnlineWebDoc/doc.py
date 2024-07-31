"""
Pegasus Workflow Management System Assistant Chat Interface

This script creates a Streamlit-based chat interface for a Pegasus workflow management system assistant. 
It uses a custom Language Model (LLM) to generate responses based on user input and information scraped 
from Pegasus documentation websites. The assistant is designed to provide information and links directly 
from the Pegasus documentation.

Key features:
1. Web scraping of Pegasus documentation
2. Custom LLM integration for response generation
3. Streamlit-based user interface for chat interaction
4. Maintenance of chat history
5. URL filtering and subdomain extraction

To run the app, use the command: streamlit run streamlit_chat.py
"""

import os
import requests
import streamlit as st
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from llm_axe import OnlineAgent, OllamaChat
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
import torch

# Streamlit interface setup
st.title("Llama Model Chat")
st.write("This app uses a fine-tuned Llama model to chat based on your input.")

# System message for the chat assistant
text = '''
        You are a Pegasus workflow management system assistant.
        You have to reply to the user from the documentation and provide the direct link of the used sections.
        Do not respond with anything else.
'''

# Commented out environment variable setting for PyTorch MPS fallback
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

st.title("Chat")
st.write("This app uses a fine-tuned Llama model to chat based on your input.")

def is_subdomain(url, main_domain):
    """Check if a URL is a subdomain of the main domain."""
    extracted_main = tldextract.extract(main_domain)
    extracted_url = tldextract.extract(url)
    return (extracted_url.domain == extracted_main.domain and
            extracted_url.suffix == extracted_main.suffix and
            extracted_url.subdomain != '')

def filter_subdomains(urls, main_domain):
    """Filter and return only subdomains from a list of URLs."""
    subdomains = set()
    for url in urls:
        if "#" in url: 
            pass
        elif is_subdomain(url, main_domain):
            subdomains.add(url)
    return subdomains

def research_internet(query):
    """Simulate internet research by extracting URLs from specific pages."""
    urls1 = extract_urls()
    urls2 = extract_urls("https://pegasus.isi.edu/documentation/reference-guide/api-reference.html")
    urls = urls1 + urls2
    return urls

def extract_urls(main_url="https://pegasus.isi.edu/documentation/"):
    """Extract URLs from a given webpage and filter for subdomains."""
    try:
        response = requests.get(main_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        anchor_tags = soup.find_all('a', href=True)
        urls = set()
        for tag in anchor_tags:
            url = urljoin(main_url, tag['href'])
            urls.add(url)
        subdomains = filter_subdomains(urls, main_url)
        return list(subdomains)
    except requests.RequestException as e:
        print(f"Error fetching {main_url}: {e}")
        return set()

# Setting up the LLM and online agent
llm = OllamaChat(model="gemma2")
online_agent = OnlineAgent(llm, custom_searcher=research_internet, temperature=0.1,)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"sender": "system", "text": text}]

# User input field
user_input = st.text_input("You:", "")

# Handle send button click
if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append({"sender": "user", "text": user_input})
        
        with st.spinner("Generating response..."):
            response = online_agent.search(user_input)
            
        st.session_state.chat_history.append({"sender": "model", "text": response})

# Display chat history with expanders
for i in range(1, len(st.session_state.chat_history), 2):
    user_message = st.session_state.chat_history[i]['text']
    model_response = st.session_state.chat_history[i+1]['text'] if i+1 < len(st.session_state.chat_history) else ""
    with st.expander(user_message):
        st.write(model_response)

# Display the system message at the top
st.write(st.session_state.chat_history[0]["text"])