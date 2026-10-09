####################  This is final file for pdf QnA chatbot   #######################


import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.documents import Document
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from streamlit_chat import message
import streamlit as st


# Set environment variable for Hugging Face API

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize Streamlit components
# st.set_page_config(
#     page_title="Category Selector",
#     layout="wide",
# )

# # Main content
st.header(":violet[My Chatbot]", divider='red')


# # Sidebar content
# with st.sidebar:
#     # Define categories and corresponding page names
#     categories = {
        
#         "Investor Centre": ["Investor"],
#         "Knowledge Centre": ["TOF", "Myths", "History"],
#     }
    
#     # Create an option menu for category selection
#     selected_category = option_menu(
#         "Select a category",
#         options=list(categories.keys()),
#         menu_icon="cast",
#         default_index=0,
#     )

#     # Check if a category is selected
#     if selected_category:
#         # Create an option menu for subcategory selection based on the selected category
#         selected_subcategory = option_menu(
#             f"Select a subcategory under {selected_category}",
#             options=categories[selected_category],
#             menu_icon="cast",
#             default_index=0,
#         )

# # Main content based on the selected category and subcategory
# if selected_category and selected_subcategory:
#     st.write(f"You selected {selected_category} > {selected_subcategory}")


###########################################################################################################################################

# option = st.selectbox(
#      r":red[$\textsf{\Large What are you looking for?}$]",
#      ('try', 'budget', 'minor'))
# st.write("") 
# st.write("")

#st.write('You selected:', option)

# st.write('Category:', option)
# st.write("")
# st.write("")
# st.write("")

# time.sleep(5)

# if(option == 'try'):
#     pdf_path = r'C:\Users\Riya\Downloads\PDFS\Bot.pdf'
#     ind='try'

# if(option == 'budget'):
#     pdf_path = r'C:\Users\Riya\Downloads\PDFS\Budget_speech.pdf'
#     ind='budget'
    
# if(option == 'minor'):
#     pdf_path = r'C:\Users\Riya\Downloads\PDFS\Minor-wise Curriculum Structures (1).pdf'  
#     ind = 'minor'
    
    
#Path to PDF file
pdf_path = r'C:\Users\Riya\Downloads\PDFS\Budget_speech.pdf'



# pdf_paths = [
#     r'C:\Users\Riya\Downloads\PDFS\Bot.pdf',
#     r'C:\Users\Riya\Downloads\PDFS\Minor-wise Curriculum Structures (1).pdf',
#     r'C:\Users\Riya\Downloads\PDFS\Faculty_Students_Major_ENR370 - Summer Internship 2024 (2).xlsx.pdf', 
# ]

# @st.cache_data
# def load_and_process_pdf(pdf_path):
#   raw_text = ''
#   for pdf_path in pdf_paths:
#       pdfreader = PdfReader(pdf_path)
#       for page in pdfreader.pages:
#           raw_text += page.extract_text()


@st.cache_data
def load_and_process_pdf(pdf_path):
    raw_text = ''
    pdfreader = PdfReader(pdf_path)
    for page in pdfreader.pages:
        raw_text += page.extract_text()
#print(raw_text)

    embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # Example model

    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=450,
      chunk_overlap=0,
      length_function=len,
)
    texts = text_splitter.split_text(raw_text)
    return embed_model, texts

    pdfreader = PdfReader(pdf_path)
    raw_text = ''.join(page.extract_text() for page in pdfreader.pages if page.extract_text())

    embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=450, chunk_overlap=0, length_function=len)
    texts = text_splitter.split_text(raw_text)


    return embed_model, texts

if pdf_path:
    embed_model, texts = load_and_process_pdf(pdf_path)


from pinecone import Pinecone
# Initialize Pinecone only if required
@st.cache_resource
def initialize_pinecone():
    api_key = "879fe352-621f-40cf-85eb-99605c1fa5d1"
    index_name = 'budget'
    pc = Pinecone(api_key=api_key)
    pc.index_name = index_name

    index_spec = {
        'name': index_name,
        'dimension': 768,  # Dimension of the embeddings
        'metric': 'cosine'
    }

    if index_name not in pc.list_indexes().names():
        pc.create_index(spec=index_spec)
    return pc.Index(index_name)

index = initialize_pinecone()

# Initialize session state variables
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context,
    and if the answer is not contained within the text below, say 'The answer is not available in the provided context.'"""
)

#The information you are trying to access is currently not avaiable. Try correct category

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")

# Initialize Hugging Face model
repo_id = "google/flan-t5-large"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 1024})

# Function to query Pinecone
def query_pinecone(query_text, top_k=3):
    query_embedding = embed_model.encode([query_text])[0].tolist()
    result = index.query(vector=query_embedding, top_k=top_k)
    return result

# Handle user query
if st.button('Submit'):
    user_query = st.session_state['input']
    st.session_state['requests'].append(user_query)

    result = query_pinecone(user_query, top_k=3)

    a = [int(''.join(filter(str.isdigit, match['id']))) for match in result['matches']]

    documents = [Document(page_content=texts[idx], metadata={}) for idx in a]

    chain = load_qa_chain(llm, chain_type="stuff")

    bot_response = chain.run(input_documents=documents, question=user_query)
    st.session_state['responses'].append(bot_response)
    #st.success(bot_response)

# Display chat history
with response_container:
    for i in range(len(st.session_state['responses'])):
        message(st.session_state['responses'][i], key=f"response_{i}")
        if i < len(st.session_state['requests']):
            message(st.session_state['requests'][i], is_user=True, key=f"request_{i}")