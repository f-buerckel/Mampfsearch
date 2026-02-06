import argparse
import csv
import logging
import sys
import os
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from mampfsearch.utils.config import (
    get_graph_storage,
    get_embedding_model,
    get_llm_client,
)
from mampfsearch.utils.schema import nodeLabels

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_lecture_transcript(lecture_name: str, graph_storage):
    """
    Fetches all segments for a lecture and combines them into one string.
    """
    logger.info(f"Fetching transcript for lecture: {lecture_name}")
    try:
        lecture_node = graph_storage.get_lecture_node(name=lecture_name)
        if not lecture_node:
            logger.error(f"Lecture '{lecture_name}' not found.")
            return None

        lecture_id = lecture_node.graph_id
        all_segments_nodes = graph_storage.get_segments_of_lecture(
            lecture_id=lecture_id
        )

        # Sort segments by position to ensure correct order
        all_segments_nodes.sort(key=lambda s: s.segment.position)

        full_transcript = "\n".join([s.segment.text for s in all_segments_nodes])
        return full_transcript
    except Exception as e:
        logger.error(f"Error fetching transcript for {lecture_name}: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple chunking by words.
    """
    words = text.split()
    chunks = []
    if not words:
        return chunks

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


class InMemoryRAG:
    def __init__(self, transcript: str):
        self.chunks = chunk_text(transcript)
        self.model = get_embedding_model()
        self.chunk_embeddings = self._embed_chunks()

    def _embed_chunks(self) -> np.ndarray:
        logger.info(f"Embedding {len(self.chunks)} chunks...")
        embeddings = []
        for chunk in self.chunks:
            output = self.model.encode(chunk, return_dense=True)
            embeddings.append(output["dense_vecs"])

        return np.array(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_emb = self.model.encode(query, return_dense=True)["dense_vecs"]

        # Cosine similarity
        # embeddings: (N, D), query: (D,)
        # sim = (A . B) / (|A| |B|)
        # Assuming embeddings are not normalized by default, we compute cosine sim manually

        scores = np.dot(self.chunk_embeddings, query_emb)
        norm_chunks = np.linalg.norm(self.chunk_embeddings, axis=1)
        norm_query = np.linalg.norm(query_emb)

        cosine_scores = scores / (norm_chunks * norm_query + 1e-10)

        top_indices = np.argsort(cosine_scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]


def generate_answer(question: str, context_chunks: List[str], llm_client) -> str:
    context_text = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant answering questions about a lecture based on the following context.\n"
        "Only rely on knowledge from the lecture and do not use your external knowledge! If you can't answer only using the lecture, say 'I cannot answer this from the provided context'.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        response = llm_client.chat.completions.create(
            model="openai/gpt-oss-20b",  # Using the model seen in existing code
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer."


def main():
    parser = argparse.ArgumentParser(
        description="Answer questions from CSV using in-memory RAG over lecture transcripts."
    )
    parser.add_argument("input_csv", help="Path to input CSV file.")
    parser.add_argument(
        "--output_csv",
        default="answered_questions.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    graph_storage = get_graph_storage()
    llm_client = get_llm_client()

    # Read questions
    questions_by_lecture = defaultdict(list)
    input_fieldnames = []
    try:
        with open(args.input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if (
                not reader.fieldnames
                or "lecture_name" not in reader.fieldnames
                or "question" not in reader.fieldnames
            ):
                logger.error("CSV must contain 'lecture_name' and 'question' columns.")
                return

            input_fieldnames = list(reader.fieldnames)
            for row in reader:
                questions_by_lecture[row["lecture_name"]].append(row)
    except FileNotFoundError:
        logger.error(f"File not found: {args.input_csv}")
        return

    # Prepare output fieldnames
    output_fieldnames = list(input_fieldnames)
    if "generated_answer" not in output_fieldnames:
        output_fieldnames.append("generated_answer")

    logger.info(f"Writing results to {args.output_csv}")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()

        for lecture_name, rows in questions_by_lecture.items():
            logger.info(f"Processing lecture: {lecture_name}")
            transcript = get_lecture_transcript(lecture_name, graph_storage)

            if not transcript:
                logger.warning(
                    f"Skipping questions for {lecture_name} due to missing transcript."
                )
                for row in rows:
                    row["generated_answer"] = "Transcript not found"
                    writer.writerow(row)
                    f.flush()
                continue

            rag_engine = InMemoryRAG(transcript)

            for row in rows:
                question = row["question"]
                logger.info(f"Answering question: {question}")

                retrieved_chunks = rag_engine.retrieve(question)
                answer = generate_answer(question, retrieved_chunks, llm_client)

                row["generated_answer"] = answer
                writer.writerow(row)
                f.flush()

    logger.info("Finished processing all questions.")


if __name__ == "__main__":
    main()
