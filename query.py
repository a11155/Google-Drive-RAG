from llama_index import StorageContext, load_index_from_storage
from dotenv import load_dotenv
from logging_spinner import SpinnerHandler
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
import logging
import streamlit as st
from initialize import readGoogleDrive
import sys
from llama_index.query_engine import RetryQueryEngine
from llama_index.evaluation import RelevancyEvaluator

def handle_user_input(query, top_k):
    print(top_k)
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k = top_k,
        node_postprocessors =  [MetadataReplacementPostProcessor(target_metadata_key="window")]
    )

    query_response_evaluator = RelevancyEvaluator()
    retry_query_engine = RetryQueryEngine(
        query_engine, query_response_evaluator
    )

    response = retry_query_engine.query(query)
    
    st.markdown(response)       
    st.markdown('----------------------------------------------')
    st.markdown('**Evidence:**')
    for i in range(len(response.source_nodes)):        
        window = response.source_nodes[i].node.metadata["window"]
        sentence = response.source_nodes[i].node.metadata["original_text"]

        st.markdown(f"**Retrieved Sentence:** {sentence}")
        st.markdown('----------------------------------------------')
        
        st.markdown(f"**Window around Sentence:** {window}") 
        st.markdown('----------------------------------------------')
        
  
    # To log the questions and answers uncomment the following lines:
    #     with open("log/q&a.csv", "a") as log:
    #         log.write(f'{question},"{response}"\n')


def main():
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(SpinnerHandler())

    load_dotenv()
    st.set_page_config(page_title="Chat with Google Drive", page_icon="🧊")

    if "index" not in st.session_state:
        st.session_state.index = None

    st.header("Chat with Google Drive :books:")
    top_k = st.sidebar.slider("Top K:", 1, 10, 4)
    user_question = st.text_input("Ask a question:")
    
    if user_question:
        if st.session_state.index:        
            handle_user_input(user_question, top_k)
        else:
            st.write("Please enter your Google Folder ID first")
    with st.sidebar:
        st.subheader("Your Google Folder ID")
        folder_id = st.text_input("Enter your Google Folder ID:")

        if st.button("Process"):

            with st.spinner("Processing"):
         
                st.session_state.index = readGoogleDrive(folder_id)

                # To use existing database uncomment the following lines
                #storage_context = StorageContext.from_defaults(persist_dir ="./storage")
                #st.session_state.index load_index_from_storage(storage_context)


    
if __name__ == '__main__':
    main()