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

from mampfsearch.utils.models import EntityCandidate, EntityRetrievalItem, Entity, ExtractionInfo
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

def chunk_text_to_sentences(text: str, max_sentences_per_chunk: int = 5) -> list[str]:
    """Chunk text into sentence groups for processing."""
    nlp_sentencizer = English()
    nlp_sentencizer.add_pipe("sentencizer")
    
    doc = nlp_sentencizer(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    for i in range(0, len(sentences), max_sentences_per_chunk):
        chunk = " ".join(sentences[i:i + max_sentences_per_chunk])
        chunks.append(chunk)
    
    return chunks

def process_pdf(pdf_file_path: str) -> list[str]: 

    pipeline_options = PdfPipelineOptions()
    #pipeline_options.do_formula_enrichment = True

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })

    result = converter.convert(pdf_file_path)

    doc = result.document
    chunker = HybridChunker()
    chunk_iter = chunker.chunk(dl_doc=doc)

    chunks = [chunk.text for chunk in chunk_iter]
    return chunks


def extract_entities(
    file_path: Path,
    max_sentences_per_chunk: int = 3,
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
        text_content = Path(file_path).read_text(encoding='utf-8')
        chunks = chunk_text_to_sentences(text_content, max_sentences_per_chunk=max_sentences_per_chunk)
        logger.debug(f"Created {len(chunks)} chunks")

    elif file_path.suffix == ".pdf":
        chunks = process_pdf(file_path)

    language = detect(" ".join(chunks[0:2]))
    logger.info(f"Detected language: {language}")

    examples_file = examples_en_path
    if language == 'de':
        examples_file = examples_de_path

    prompt = Path(prompt_path).read_text(encoding='utf-8')
    
    nlp_llm = assemble(config_path, overrides={"paths.examples": str(examples_file),
                                           "components.llm.task.template": str(prompt)})

    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        
        # temporary fix for: https://github.com/vllm-project/vllm/issues/22403
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                doc = nlp_llm(chunk)
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
            )

            new_num, merged_num = insert_entity_candidate(entity_candidate)
            num_new_inserted_entities += new_num
            num_merged_entities += merged_num
                
        
        logger.debug(f"Found {len(chunk_entities)} entities: {chunk_entities}")
        if print_chunks:
            logger.info(f"Chunk text:\n{chunk}")
            logger.info(f"Entities in chunk {i+1}:")
            for entity in chunk_entities:
                logger.info(f"{entity[0]} : {entity[1]}")

        logger.info(50*"-")
    
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