import logging

from spacy import Language
from mampfsearch.utils import config, prompts


logger = logging.getLogger(__name__)


@Language.factory(
    "llm_ner_validation",
    requires=["doc.ents"],
    assigns=["doc.ents"],
)
class LLM_NER_VALIDATION:
    def __init__(self, nlp, name):
        self.llm_client = config.get_llm_client()

    def __call__(self, doc):
        if not doc.ents:
            return doc

        validated_ents = self.validate_entities(list(doc.ents))
        doc.ents = validated_ents

        return doc

    def validate_entities(self, ents):
        validated = []

        for ent in ents:
            if self.validate_entity(ent):
                validated.append(ent)

        return validated

    def validate_entity(self, ent) -> bool:
        prompt = prompts.NER_VALIDATION_PROMPT.format(entity=ent.text)

        try:
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
            )
            content = response.choices[0].message.content.lower()

            if "yes" in content:
                logger.debug(
                    f"Entity '{ent.text}' with label '{ent.label_}' validated by LLM."
                )
                return True
            else:
                logger.info(
                    f"Entity '{ent.text}' with label '{ent.label_}' rejected by LLM."
                )
                return False
        except Exception as e:
            logger.error(f"Error during entity validation LLM call: {e}")
            return True
