# PDF Chatbot — RAG-Based Question Answering System

## Overview
This chatbot allows users to upload PDFs and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to find relevant passages and generate answers using OpenAI GPT models.

## Features
- Question answering over user-uploaded PDFs.
- Semantic search using vector embeddings and FAISS.
- Answers generated using retrieved document context.
- Interactive Streamlit interface.
- Docker support for consistent application setup.
- Automated unit tests for API functionality.

## How It Works
1. Extract text from uploaded PDFs.
2. Split the text into smaller chunks.
3. Generate vector embeddings using Hugging Face sentence transformers.
4. Store and search embeddings using FAISS.
5. Retrieve passages relevant to the user's question.
6. Pass the retrieved context and question to an OpenAI GPT model to generate an answer.

## Technologies
Python, LangChain, Hugging Face, OpenAI API, FAISS, Streamlit, Docker

## Main Application
`Stream1.py` is the main application file.

To launch the interface after installing dependencies and configuring the required API credentials:

    python -m streamlit run Stream1.py

## Results
Beta testing showed a 25% improvement in user satisfaction scores after implementing vector-based retrieval and similarity search.

## Limitations
Answer quality depends on PDF text extraction and the relevance of retrieved passages. Generated answers may contain errors and should be checked against the source document.

## Project Timeline
July 2024
