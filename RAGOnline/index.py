import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

       
# 3. Call Ollama Llama3 model
def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']
    

def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_context = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_context)

st.title("Chat with Pegasus 🌐")
st.caption("This app allows you to chat with a pegasus documentation using local Llama-3 and RAG")

# Get the webpage URL from the user
webpage_url = st.text_input("Enter Webpage URL", type="default")
def extract_urls(main_url):
    try:
        # Send a GET request to the main URL
        response = requests.get(main_url)
        response.raise_for_status()  # Check if the request was successful

        # Parse the content of the page with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all anchor tags
        anchor_tags = soup.find_all('a', href=True)

        # Extract and join URLs
        urls = set()
        for tag in anchor_tags:
            url = urljoin(main_url, tag['href'])
            urls.add(url)

        return urls
    except requests.RequestException as e:
        print(f"Error fetching {main_url}: {e}")
        return set()

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def set_clicked():
    st.session_state.clicked = True

def is_subdomain(url, main_domain):
    extracted_main = tldextract.extract(main_domain)
    extracted_url = tldextract.extract(url)

    return (extracted_url.domain == extracted_main.domain and
            extracted_url.suffix == extracted_main.suffix and
            extracted_url.subdomain != '')

def filter_subdomains(urls, main_domain):
    subdomains = set()
    for url in urls:
        if is_subdomain(url, main_domain):
            subdomains.add(url)
    return subdomains
selected_urls = []

st.button("Extract Subdomains",on_click=set_clicked)

if st.session_state.clicked:
    # Extract URLs
    urls = extract_urls(webpage_url)
    
    # Filter subdomains
    subdomains = filter_subdomains(urls, webpage_url)
    
    # Display the extracted subdomain URLs as a checklist
    if urls:
        st.subheader("Extracted URLs")
        urls_updated=urls.copy()
        urls_updated.add("*")
        selected_urls= st.multiselect("Select Subdomains to include in RAG",  urls_updated, default="*")
        
if st.button("Show selected  Subdomains"):
        if len(selected_urls)==0:
             st.write("No url selected or url provided")
        elif "*" in  selected_urls:
                selected_urls = []
                selected_urls=urls
                st.write("You selected all urls")
        else: 
            st.write("You selected:")
            for url in selected_urls:
                st.write(url)
st.button("RAG it  and chat with",on_click=set_clicked)
if st.session_state.clicked:
    if "*" in  selected_urls:
                selected_urls = []
                selected_urls=list(urls)
    loader = WebBaseLoader(selected_urls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    # 2. Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model="gemma2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    #4. RAG Setup
    retriever = vectorstore.as_retriever()
    # Ask a question about the webpage
    prompt = st.text_input("Ask any question about the pegasus doc")

    # Chat with the webpage
    if prompt:
        result = rag_chain(prompt)
        st.write(result)
    #
    st.success(f"Loaded sub-domaine of {webpage_url} successfully!")

