# from langchain_core.documents import Document
# import streamlit as st
# from streamlit_chat import message
# from PyPDF2 import PdfReader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.chains.question_answering import load_qa_chain
# import pinecone
# import os
# from langchain.text_splitter import CharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # pdf_paths = [
# #     # r'C:\Users\Riya\Downloads\PDFS\Bot.pdf',
# #     # r'C:\Users\Riya\Downloads\PDFS\Budget_speech.pdf',
# #     # r'C:\Users\Riya\Downloads\PDFS\Minor-wise Curriculum Structures (1).pdf',
# #     # r'C:\Users\Riya\Downloads\PDFS\Faculty_Students_Major_ENR370 - Summer Internship 2024 (2).xlsx.pdf',
# #     r'C:\Users\Riya\Downloads\PDFS\SEBI_Monthly_Portfolio 31 May 2024 - Sheet1.csv',
    
# # ]

# import csv

# from langchain.document_loaders import CSVLoader
# loader=CSVLoader('SEBI_Monthly_Portfolio 31 May 2024 - Sheet1.csv')
# # List of paths to your CSV files
# #csv_paths = ["SEBI_Monthly_Portfolio 31 May 2024 - Sheet1.csv"]  # Replace with your CSV file paths

# # Initialize an empty string to store the concatenated text
# raw_text = ''

# # Loop through each CSV file
# for csv_path in csv_paths:
#     with open(csv_path, 'r', encoding='utf-8') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         next(csv_reader)  # Skip header if exists
        
#         # Concatenate text from each row
#         for row in csv_reader:
#             # Assuming the text is in the first column
#             raw_text += row[0] + '\n'  # Add newline to separate rows

# # Print or use the concatenated text
# #print(raw_text)



# # for single file 

# # pdf_path= r'C:\Users\Riya\Downloads\PDFS\Minor-wise Curriculum Structures (1).pdf'
# # pdfreader = PdfReader(pdf_path)
# # raw_text = ''
# # for i, page in enumerate(pdfreader.pages):
# #     content = page.extract_text()
# #     if content:
# #         raw_text += content

# # raw_text = ''
# # for pdf_path in pdf_paths:
# #     pdfreader = PdfReader(pdf_path)
# #     for page in pdfreader.pages:
# #         raw_text += page.extract_text()

# #print(raw_text)

# embed_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')  # Example model

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=10,
#     chunk_overlap=2,
#     length_function=len,
# )
# texts = text_splitter.split_text(raw_text)
# #print(texts)

# embeddings_list = []
# for text in texts:
#     embeddings = embed_model.encode(text).tolist()
#     embeddings_list.append(embeddings)

# print(len(embeddings_list))

# from pinecone import Pinecone

# api_key = "879fe352-621f-40cf-85eb-99605c1fa5d1"
# index_name = "all"
# pc = Pinecone(api_key=api_key)
# pc.index_name = index_name

# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=len(embeddings_list[0]),
#         metric='cosine'
#     )

# index = pc.Index(index_name)

# vectors = [{"id": f"pdf_chunk_{i}", "values": embeddings_list[i]} for i in range(len(embeddings_list))]
# index.upsert(vectors)

# print("Embeddings have been successfully uploaded to Pinecone.")


from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from langchain_community.document_loaders import CSVLoader

# Load documents from CSV
loader = CSVLoader(file_path=r"C:\Users\Riya\Downloads\PDFS\SEBI_Monthly_Portfolio 31 May 2024 - Sheet1.csv")
docu = loader.load()

# Initialize the embedding model
embed_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
  chunk_size=450,
  chunk_overlap=0
)

# Split documents into chunks of text
texts = text_splitter.split_documents(docu)

# Extract the text content from each chunk and encode it
embeddings_list = []
for text in texts:
    text_content = text.page_content  # Access the text content correctly
    embeddings = embed_model.encode(text_content).tolist()
    embeddings_list.append(embeddings)
    #print(embeddings)

#print(len(embeddings_list))


from pinecone import Pinecone

api_key = "879fe352-621f-40cf-85eb-99605c1fa5d1"
index_name = "all"
pc = Pinecone(api_key=api_key)
pc.index_name = index_name

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=len(embeddings_list[0]),
        metric='cosine'
    )

index = pc.Index(index_name)

vectors = [{"id": f"pdf_chunk_{i}", "values": embeddings_list[i]} for i in range(len(embeddings_list))]
index.upsert(vectors)
print(vectors)
print("Embeddings have been successfully uploaded to Pinecone.")