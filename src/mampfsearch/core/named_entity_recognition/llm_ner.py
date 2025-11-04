import logging
import os
from pathlib import Path
from typing import List

from spacy_llm.util import assemble

from mampfsearch.utils.models import EntityCandidate, Chunk
from . import BaseNER

logger = logging.getLogger(__name__)


class LLM_NER(BaseNER):
    
    def __init__(self, language: str = "en", retry_attempts: int = 3):
        self.language = language
        self.retry_attempts = retry_attempts
        self.nlp_llm = self._initialize_model()
    
    def _initialize_model(self):
        file_dir = Path(__file__).parent
        config_path = file_dir / "ner_config.cfg"
        prompt_path = file_dir / "ner_prompt.txt"
        
        examples_file = (
            file_dir / "math_examples_de.json" 
            if self.language == "de" 
            else file_dir / "math_examples_en.json"
        )
        
        prompt = prompt_path.read_text(encoding='utf-8')
        
        return assemble(
            str(config_path),
            overrides={
                "paths.examples": str(examples_file),
                "components.llm.task.template": prompt
            }
        )
    
    def extract(self, chunk: Chunk) -> List[EntityCandidate]:
        
        for attempt in range(self.retry_attempts):
            try:
                doc = self.nlp_llm(chunk.text)
                
                entities = [
                    EntityCandidate(
                        text=ent.text.lower(),
                        label=ent.label_,
                        Location=chunk.location
                    )
                    for ent in doc.ents
                ]
                
                return entities
                
            except Exception as e:
                logger.warning(
                    f"Entity extraction failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                if attempt == self.retry_attempts - 1:
                    logger.error(
                        f"Entity extraction failed after {self.retry_attempts} retries"
                    )
                    return []
        
        return []