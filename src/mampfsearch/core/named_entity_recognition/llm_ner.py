import logging

from pathlib import Path

from mampfsearch.utils import config, prompts

from spacy import Language
from spacy_llm.util import assemble
from spacy.tokens import Span

logger = logging.getLogger(__name__)

@Language.factory(
    "llm_ner_v2",
    default_config={"retry_attempts": 3},
    assigns=["doc.ents"],
    )
class LLM_NER():
    
    def __init__(self, nlp, name, retry_attempts: int = 3):
        self.language = nlp.lang
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
    
    def __call__(self, doc):
        for attempt in range(self.retry_attempts):
            try:
                llm_doc = self.nlp_llm(doc.text)

                # have to rebuild the entities
                new_ents = []
                for ent in llm_doc.ents:
                    # Create new Span with original doc's vocab
                    span = Span(doc, ent.start, ent.end, label=ent.label_)
                    new_ents.append(span)
                
                doc.ents = new_ents
                logger.info(f"LLM NER extracted the following entities: {[ent.text for ent in doc.ents]}")
                return doc
            except Exception as e:
                logger.warning(f"LLM NER call failed on attempt {attempt + 1}/{self.retry_attempts}: {e}")

        return doc
    
