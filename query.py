from llama_index import StorageContext, load_index_from_storage
from dotenv import load_dotenv
from logging_spinner import SpinnerHandler
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
import logging
import streamlit as st
from initialize import readGoogleDrive
import sys


def handle_user_input(query):
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k = 2,
        node_postprocessors =  [MetadataReplacementPostProcessor(target_metadata_key="window")]
    )


    response = query_engine.query(query)
    
    st.write(response)        
    
  

   #     with open("log/q&a.csv", "a") as log:
  #          log.write(f'{question},"{response}"\n')


def main():
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(SpinnerHandler())

    load_dotenv()
    st.set_page_config(page_title="Chat with Google Drive", page_icon="🧊")

    if "index" not in st.session_state:
        st.session_state.index = None

    st.header("Chat with Google Drive :books:")

    user_question = st.text_input("Ask a question:")
    
    if user_question:
        if st.session_state.index:        
            handle_user_input(user_question)
        else:
            st.write("Please enter your Google Drive ID first")
    with st.sidebar:
        st.subheader("Your Google Drive ID")
        folder_id = st.text_input("Enter your Google Drive ID:")

        if st.button("Process"):

            with st.spinner("Processing"):
                #storage_context = StorageContext.from_defaults(persist_dir ="./storage")
                #load_index_from_storage(storage_context)
                st.session_state.index = readGoogleDrive(folder_id)


    
if __name__ == '__main__':
    main()