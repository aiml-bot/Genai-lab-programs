Experiment No.10: 
Build a chat bot for the Indian Penal Code. We'll start by downloading the official Indian 
Penal Code document, and then we'll create a chat bot that can interact with it. Users will be 
able to ask questions about the Indian Penal Code and have a conversation with it.
# First, install the required package
!pip install PyMuPDF

# Now import the modules
import os
import requests
import numpy as np
import faiss
import fitz  # PyMuPDF

from langchain.llms import Cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Step 2: Load and extract text from PDF
pdf_path = "ipc.pdf"
pdf_document = fitz.open(pdf_path)

ipc_text = ""
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    ipc_text += page.get_text()

with open("IPC_text.txt", 'w', encoding="utf-8") as text_file:
    text_file.write(ipc_text)

print("Text extracted and saved!")

# Step 3: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(ipc_text)

# Step 4: Create embeddings and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(texts, convert_to_tensor=True)

index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings.cpu().numpy())

# Step 5: Setup Cohere API
os.environ["COHERE_API_KEY"] = "YrqBdTypjvdKMc7bB1jLwihs5TS54JCN8qjrLVQS"
llm = Cohere(model="command-xlarge-nightly", temperature=0.7)

# Step 6: Define function to get a response from the chatbot
def get_chat_response(user_query):
    query_embedding = model.encode([user_query], convert_to_tensor=True)
    D, I = index.search(query_embedding.cpu().numpy(), k=1)
    most_similar_text = texts[I[0][0]]

    prompt = f"""
    The user has asked a question related to the Indian Penal Code. Below is the relevant section from the Indian Penal Code:

    {most_similar_text}

    The user's question: {user_query}

    Please provide an answer based on the above IPC section.
    """

    response = llm(prompt)
    return response

# Step 7: Get user input and display response
user_input = input("Ask a question about the Indian Penal Code: ")
response = get_chat_response(user_input)
print(f"Chatbot Response: {response}")
