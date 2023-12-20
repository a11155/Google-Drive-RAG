from dotenv import load_dotenv
from llama_index.embeddings import HuggingFaceEmbedding
from logging_spinner import SpinnerHandler
from llama_index.node_parser import SimpleNodeParser
from sentence_transformers import SentenceTransformer
from pathlib import Path
from llama_index.evaluation.eval_utils import get_responses
from llama_index.evaluation import BatchEvalRunner
from llama_index import ServiceContext, set_global_service_context
from llama_index import StorageContext, load_index_from_storage
import asyncio
import numpy as np
from llama_index.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    DatasetGenerator,
    QueryResponseDataset,
)
import random

from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser

import logging
import os


load_dotenv()



node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=6, # one large document vs many little documents
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)


simple_node_parser = SimpleNodeParser.from_defaults()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
embed_model_base = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
)

ctx = ServiceContext.from_defaults(
    llm=llm
)



def test(query_engine):

    
    window_response = query_engine.query("What are the objectives?")

    print(window_response.response)

    window = window_response.source_nodes[0].node.metadata["window"]
    sentence = window_response.source_nodes[0].node.metadata["original_text"]


    print(f"Window: {window}")
    print("-----------------------------")
    print(f"Sentence: {sentence}")

async def get_batch_eval_results(batch_runner, queries, pred_responses, ref_response_strs, index):
    return await batch_runner.aevaluate_responses(
        queries=queries,
        responses=pred_responses,
        reference=ref_response_strs,
    )
async def get_eval_dataset(nodes, eval_service_ctx = ctx):
    num_nodes_eval = 10
    sample_eval_nodes = random.sample(nodes, num_nodes_eval)
    dataset_generator = DatasetGenerator(
        sample_eval_nodes,
        service_context = eval_service_ctx,
        show_progress = True,
        num_questions_per_chunk = 2,
    )
    eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()
    eval_dataset.save_json("eval_qr_dataset.json")


def get_nodes(documents, node_parser):
    nodes = node_parser.get_nodes_from_documents(documents)

    return nodes

def evaluate_st(dataset, model_id, name):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationretrievalEvaluator(queries, corpus, relevant_docs, name =name)
    
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    return evaluator(model, output_path=output_path)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(SpinnerHandler())

storage_context = StorageContext.from_defaults(persist_dir ="./storage")


logger.info('Loading data...', extra={'user_waiting': True})
sentence_index = load_index_from_storage(storage_context)
logger.info('Finished Loading!', extra={'user_waiting': False})

query_engine = sentence_index.as_query_engine(
    similarity_top_k = 3, node_postprocessors =  [MetadataReplacementPostProcessor(target_metadata_key="window")]
)

eval_service_ctx  = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))

#asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#asyncio.run(get_eval_dataset(nodes, eval_service_ctx))

eval_dataset = QueryResponseDataset.from_json("eval_qr_dataset.json")


evaluator_c = CorrectnessEvaluator(service_context = eval_service_ctx)
evaluator_s = SemanticSimilarityEvaluator(service_context = eval_service_ctx)
evaluator_r = RelevancyEvaluator(service_context = eval_service_ctx)
evaluator_f = FaithfulnessEvaluator(service_context = eval_service_ctx)

max_samples = 5

eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]


query_engine = sentence_index.as_query_engine(
    similarity_top_k = 3, node_postprocessors =  [MetadataReplacementPostProcessor(target_metadata_key="window")]
)
pred_responses = get_responses(eval_qs[:max_samples],  query_engine, show_progress=True)

pred_responses_strs = [p for p in pred_responses]


evaluator_dict = {
    "correctness": evaluator_c,
    
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
batch_runner = BatchEvalRunner(evaluator_dict, show_progress=True, workers = 2)

#results = asyncio.run(get_batch_eval_results(batch_runner, eval_qs[:max_samples], pred_responses_strs[:max_samples], ref_response_strs[:max_samples]))

print(len(eval_qs[:max_samples]), len(pred_responses_strs[:max_samples]), len(ref_response_strs[:max_samples]))

results = loop.run_until_complete(get_batch_eval_results(batch_runner, eval_qs[:max_samples], pred_responses_strs[:max_samples], ref_response_strs[:max_samples], sentence_index.as_query_engine()))

print(results)

results_df = get_results_df([
     results],
     ["GPT-3"],
    ["Correctness", "Semantic Similarity", "Relevancy", "Faithfulness"],
)

print(results_df)


#dataset = generate_qa_embedding_pairs(nodes)

#print(evaluate_st(dataset, "BAAI/bge-small-en-v1.5", "initial"))



#index = VectorStoreIndex.from_documents(documents)
#index.storage_context.persist()