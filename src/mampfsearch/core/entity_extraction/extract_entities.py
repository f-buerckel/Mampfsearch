import spacy
import time
import logging
import os
import uuid

from pathlib import Path
from qdrant_client.models import PointStruct
from langdetect import detect
from enum import Enum
from pathlib import Path
from collections import Counter
from typing import Union, Optional

from mampfsearch.core.chunking import chunk_text_by_sentences, chunk_pdf_file, chunk_srt_file
from mampfsearch.utils.models import EntityCandidate, EntityRetrievalItem, Entity, ExtractionInfo, Chunk, VideoLocation, FileLocation
from mampfsearch.utils import config
from mampfsearch.retrievers import EntityRetriever

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

from spacy.training import Example
from spacy.tokens import Doc
from spacy.lang.en import English
from spacy_llm.util import assemble


logger = logging.getLogger(__name__)

def extract_entities(
    file_path: Path,
    course_id: str,
    lecture_id: Optional[str] = None,
    print_chunks: bool = False
) -> ExtractionInfo:

    num_extracted_entities = 0
    num_new_inserted_entities = 0
    num_merged_entities = 0
    
    file_dir = os.path.dirname(__file__)
    config_path = os.path.join(file_dir, "ner_config.cfg")
    prompt_path = os.path.join(file_dir, "ner_prompt.txt")
    examples_en_path = os.path.join(file_dir, "math_examples_en.json")
    examples_de_path = os.path.join(file_dir, "math_examples_de.json")

    chunks = []
    if file_path.suffix == ".txt":
        max_sentences_per_chunk = 3
        text_content = Path(file_path).read_text(encoding='utf-8')
        chunks = chunk_text_by_sentences(
            text=text_content,
            course_id=course_id,
            max_sentences_per_chunk=max_sentences_per_chunk,
        )

    elif file_path.suffix == ".pdf":
        chunks = chunk_pdf_file(
            pdf_file_path=file_path,
            course_id=course_id,
            enable_formula_enrichment=True
        )
    
    elif file_path.suffix == ".srt":
        # the high max_chunk_size encouages to keep sentences in one chunk.
        min_chunk_size = 40
        max_chunk_size = 400
        chunks = chunk_srt_file(
            srt_file=file_path,
            course_id=course_id,
            lecture_id=lecture_id if lecture_id else "unknown",
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

    language = detect(" ".join([chunk.text for chunk in chunks[0:2]]))
    logger.info(f"Detected language: {language}")

    examples_file = examples_en_path
    if language == 'de':
        examples_file = examples_de_path

    prompt = Path(prompt_path).read_text(encoding='utf-8')
    
    nlp_llm = assemble(config_path, overrides={"paths.examples": str(examples_file),
                                           "components.llm.task.template": str(prompt)})

    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk.text.split())} words)")
        
        # temporary fix for: https://github.com/vllm-project/vllm/issues/22403
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                doc = nlp_llm(chunk.text)
            except Exception as e:
                logger.warning(f"Entity extraction failed for chunk {i} (attempt {attempt}/{retry_attempts}): {e}")
            else: 
                break
        else:
            logger.warning(f"Entitiy extraction failed for chunk {i} after {retry_attempts} retries.")
            continue

        chunk_entities = [(ent.text, ent.label_) 
                        for ent in doc.ents]
        num_extracted_entities += len(chunk_entities)

        for ent in doc.ents:
            entity_candidate = EntityCandidate(
                text = ent.text.lower(),
                label = ent.label_,
                Location = chunk.location
            )

            is_new, is_merged = insert_entity_candidate(entity_candidate)
            num_new_inserted_entities += is_new
            num_merged_entities += is_merged
                
        
        logger.debug(f"Found {len(chunk_entities)} entities: {chunk_entities}")
        if print_chunks:
            logger.info(f"Chunk text:\n{chunk}")
            logger.info(f"Entities in chunk {i+1}:")
            for entity in chunk_entities:
                logger.info(f"{entity[0]} : {entity[1]}")

        logger.info(50*"-")
    
    logger.info(f"Extraction complete. Extracted {num_extracted_entities} entities.")
    logger.info(f"Inserted {num_new_inserted_entities} new entities, merged {num_merged_entities} existing entities.")
    
    return ExtractionInfo(
        num_extracted_entities=num_extracted_entities,
        num_new_inserted_entities=num_new_inserted_entities,
        num_merged_entities=num_merged_entities
    )

def insert_entity(entity_candidate: EntityCandidate):

    model = config.get_embedding_model()
    entity_text = entity_candidate.text

    embedding = model.encode(entity_text, return_dense=True)
    payload = Entity.from_entity_candidate(entity_candidate).model_dump()

    logger.debug(f"Inserting entity '{entity_candidate.text}')")

    qdrant_client = config.get_qdrant_client()
    qdrant_client.upsert(
        collection_name=config.ENTITIES_COLLECTION_NAME,
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                payload = payload,
                vector = {
                    "dense": embedding["dense_vecs"],
                }
            )
        ]
    )

    return


def insert_entity_candidate(
    entity_candidate: EntityCandidate,
    ):
    # returns whether a new entity was inserted or an existing entity was merged: (is_new, is_merged)

    retriever = EntityRetriever()

    results = retriever.retrieve(entity_candidate.text, limit=1)

    if not results or results[0].score < config.ENTITY_EMBED_SIM_THRESHOLD:
        logger.info(f"Inserting new entity '{entity_candidate.text}' with label '{entity_candidate.label}'")
        insert_entity(entity_candidate)
        return (1, 0)

    else:
        logger.info(f"Entity '{entity_candidate.text}' already in knowledge base with name {results[0].entity.name} (score: {results[0].score})")
        merge_entities(entity_candidate, results[0])
        return (0, 1)
    
    return (0, 0)

def merge_entities(
    entity_candidate: EntityCandidate,
    entity_kb: EntityRetrievalItem,
    ):

    entity = entity_kb.entity
    entity_instances = entity.entity_instances or []

    labels = [ent.label for ent in entity_instances] + [entity_candidate.label]
    most_common_label = Counter(labels).most_common(1)[0][0] # has form [("Theorem", 10)]

    aliases = [ent.text for ent in entity_instances] + [entity_candidate.text]
    most_common_name = Counter(aliases).most_common(1)[0][0] # has form [("backpropagation", 10)]

    if most_common_label != entity.label:
        logger.info(f"Updating entity '{entity.name}' label from '{entity.label}' to '{most_common_label}'")
        entity.label = most_common_label

    if most_common_name != entity.name:
        logger.info(f"Updating entity '{entity.name}' name from '{entity.name}' to '{most_common_name}'")
        entity.name = most_common_name
    
    entity_instances.append(entity_candidate)

    id = entity_kb.id

    client = config.get_qdrant_client()
    client.set_payload(
        collection_name=config.ENTITIES_COLLECTION_NAME,
        payload = {
            "name": entity.name,
            "entity_instances": [entity_instance.model_dump() for entity_instance in entity_instances],
            "label": entity.label,
        },
        points=[id],
    )

    return