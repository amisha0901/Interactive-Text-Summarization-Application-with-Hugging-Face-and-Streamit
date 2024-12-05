import streamlit as st
import os
from langchain import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Setting the API key
os.environ['API_KEY'] = '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'

def generate_response(txt):
    falcon_llm = HuggingFaceHub(
        huggingfacehub_api_token=os.environ['API_KEY'],
        repo_id='tiiuae/falcon-7b-instruct',
        model_kwargs={'temperature': 0.6, 'max_new_tokens': 100}
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    
    # Text summarization
    chain = load_summarize_chain(falcon_llm, chain_type='map_reduce')
    return chain.run(docs)

# Page title
st.set_page_config(page_title='Text Summarization App')
st.title('Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('AI is Thinking...'):
            response = generate_response(txt_input)
            result.append(response)

if result:
    st.info(result[0])  # Displaying the first result assuming there's only one response
