from llama_index import download_loader
from dotenv import load_dotenv
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex
from logging_spinner import SpinnerHandler
import logging
import os


from googleDriveReader import GoogleDriveReader
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(SpinnerHandler())


folder_id = os.getenv('FOLDER_ID')

loader = GoogleDriveReader()

logger.info('Loading data...', extra={'user_waiting': True})
documents = loader.load_data(folder_id=folder_id)
logger.info('Finished loading!', extra={'user_waiting': False})


index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()