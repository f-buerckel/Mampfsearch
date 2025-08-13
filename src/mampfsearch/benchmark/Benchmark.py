from .EvaluationDataset import EvaluationDataset
from mampfsearch.retrievers import BaseRetriever
from mampfsearch.commands.lectures.init import create_lectures_collection
from mampfsearch.commands.lectures.insert_srt import insert_srt
import logging
import time
import math

class Benchmark:

    def __init__(self, eval_dataset: EvaluationDataset, retriever: BaseRetriever, name):
        self.eval_dataset = eval_dataset
        self.retriever = retriever
        self.collection_name = name + "benchmark"
        self.logger = logging.getLogger(__name__)
    
    def _init_collection(self):
        # only insert srt if collection does not exist
        if create_lectures_collection(name=self.collection_name) == 0:
            insert_srt(
                lecture_name=self.eval_dataset.data["config"]["lecture_name"],
                lecture_position=0,
                srt_file=self.eval_dataset.absolute_srt_path,
                collection_name=self.collection_name
            )
    
    def run(self):
        self._init_collection()

        start_time = time.time()
        self.logger.info(f"Running benchmark on collection '{self.collection_name}' with {len(self.eval_dataset)} questions.")

        score_sum = 0.0
        for i in range(len(self.eval_dataset)):
            question_text = self.eval_dataset.get_question_text(i)
            relevant_documents = self.eval_dataset.get_relevant_documents(i)

            retrieved_documents = self.retriever.retrieve(question_text, self.collection_name, limit=len(relevant_documents))
            # Convert results to a dictionary with same structure as relevant_documents
            retrieved_documents = {result.position: result.score for result in retrieved_documents}

            score = self._evaluate_question(relevant_documents, retrieved_documents)
            score_sum += score
            self.logger.info(f"Question {i}: {question_text} \n Score: {score:.4f}")
            self.logger.info(f"Relevant documents: {relevant_documents}")
            self.logger.info(f"Retrieved documents: {retrieved_documents}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        average_score = score_sum / len(self.eval_dataset) if len(self.eval_dataset) > 0 else 0.0
        self.logger.info(f"\nAverage score for benchmark '{self.collection_name}': {average_score:.4f}")
        self.logger.info(f"Benchmark completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
        self.logger.info(50*"-")

        return {
            'average_score': average_score,
            'duration_seconds': duration,
            'time_per_question': duration / len(self.eval_dataset)
        }
    
    def _evaluate_question(self, relevant_documents: dict, retrieved_documents: dict) -> float:
        """
        Evaluates a question using a retriever and comparing against the manually selected relevant documents.
        """

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_documents):
            # relevance is 0 if the document does not appear in the manually selected relevant_documents
            relevance = relevant_documents.get(doc_id, 0.0)
            dcg += (2**relevance - 1) / math.log2(i + 2)
        
        # Calculate ideal dcg to get normalized result
        ideal_dcg = 0.0
        for i, relevance in enumerate(relevant_documents.values()):
            ideal_dcg += (2**relevance - 1) / math.log2(i + 2)
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    


