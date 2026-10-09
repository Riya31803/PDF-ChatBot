import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, hf_hub_download
import fitz  # PyMuPDF
import os

# Authenticate with Hugging Face
def authenticate_huggingface(token):
    try:
        login(token=token)
        st.success("Authenticated with Hugging Face successfully!")
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.stop()

# Download model and tokenizer
def download_model(model_name):
    try:
        tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.stop()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf_document:
            for page in pdf_document:
                pdf_text += page.get_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return pdf_text

# Generate an answer to the question based on context
def generate_answer(question, context, tokenizer, model):
    try:
        inputs = tokenizer(question + " " + context, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs["input_ids"], max_length=512, num_return_sequences=1)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer."

def main():
    st.title("PDF Question Answering Chatbot")

    # Authentication
    token = st.text_input("Enter your Hugging Face token:")
    if token:
        authenticate_huggingface(token)

        # Model Download
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer, model = download_model(model_name)

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            with st.spinner("Extracting text..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                st.success("Text extraction complete!")
                st.text_area("PDF Content", pdf_text, height=300)
            
            question = st.text_input("Ask a question about the PDF:")
            if question:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, pdf_text, tokenizer, model)
                    st.write("*Answer:*", answer)

if __name__ == "_main_":
    main()
    
    
    