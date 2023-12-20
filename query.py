from llama_index import StorageContext, load_index_from_storage
from dotenv import load_dotenv
from logging_spinner import SpinnerHandler
import logging
import sys


load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(SpinnerHandler())


storage_context = StorageContext.from_defaults(persist_dir ="./storage")



logger.info('Loading data...', extra={'user_waiting': True})
index = load_index_from_storage(storage_context)
logger.info('Finished Loading!', extra={'user_waiting': False})
while True:
    question = input("Ask a question: ")


    logger.info('Generating Response...', extra={'user_waiting': True})
    response = index.as_query_engine().query(question)
    logger.info(response, extra={'user_waiting': False})
    

    log_question = input("Log Question? (y/n): ")

    if log_question != "y":
        continue

    with open("log/q&a.csv", "a") as log:
        log.write(f'{question},"{response}"\n')