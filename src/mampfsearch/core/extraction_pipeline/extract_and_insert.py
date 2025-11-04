import logging

from pathlib import Path
from langdetect import detect
from typing import Optional

from mampfsearch.utils.models import ExtractionInfo
from mampfsearch.core.named_entity_recognition import LLM_NER
from mampfsearch.core.entity_linking import EmbeddingEntityLinker
from mampfsearch.core.chunking import chunk_file

logger = logging.getLogger(__name__)

def extract_and_insert(
    file_path: Path,
    course_id: str,
    lecture_id: Optional[str] = None,
    print_chunks: bool = False
) -> ExtractionInfo:

    num_extracted_entities = 0
    num_new_inserted_entities = 0
    num_merged_entities = 0

    chunks = chunk_file(
        file_path=file_path,
        course_id=course_id,
        lecture_id=lecture_id,
    )

    language = detect(" ".join([chunk.text for chunk in chunks[0:2]]))
    logger.info(f"Detected language: {language}")

    ner = LLM_NER(language=language)
    linker = EmbeddingEntityLinker()
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk.text.split())} words)")
        
        entities = ner.extract(chunk)
        
        chunk_entities = [(ent.text, ent.label) for ent in entities]
        num_extracted_entities += len(chunk_entities)

        for entity_candidate in entities:
            is_new, is_merged = linker.link(entity_candidate)
            num_new_inserted_entities += int(is_new)
            num_merged_entities += int(is_merged)
                
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