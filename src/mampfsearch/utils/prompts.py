SEMANTIC_SEARCH_PROMPT = """ Retrieve semantically similar text. Search term: {term}"""
QUESTION_SEARCH_PROMPT = """ Given a math question, retrieve lecture passages that are relevant to the question. Question: {question}"""
QA_PROMPT = """
You are a math-aware assistant.
Your primary instruction is to **answer the question in the same language as the original 'Question' below, while only using the information from the 'context' given.**
Use the following pieces of retrieved context to answer the question.
If the answer CANNOT be extracted verbatim from the provided context, reply exactly: I don't know.
Keep your answer under three sentences and concise.  
Question: {question} 
Context: {context} 
"""
RAG_PROMPT_JSON = """You are an AI assistant that answers questions based on lecture transcripts. Based on the given context provide an accurate and concise answer to the question.

INSTRUCTIONS:
1. Answer the question using the provided source document. You may rephrase the information to shorten the answer. Only add context if you are extremely confident in the added context and relevance.
2. If information is insufficient, state "I don't know" 
3. Provide a confidence score (0.0-1.0) where:
   - 1.0 = Answer directly stated in context
   - 0.7-0.9 = Answer can be reasonably inferred
   - 0.3-0.6 = Partial information available
   - 0.0-0.2 = Very limited or unclear information
4. Extract Verbatim Snippets: From the source documents, extract short, verbatim snippets that directly support your answer.
    - RULE: Snippets MUST be short and precise. A snippet should ideally be a single sentence or a shortened part of one.
    - RULE: DO NOT extract long paragraphs or multiple sentences as a single snippet.
    - Use '[...]' to indicate omitted text within a sentence. For example: "LDA [...] is a linear classifier."
    - Provide a relevance score from 0 to 1 for each snippet.

SOURCE DOCUMENTS:
{context}

QUESTION: {question}

Respond in the following JSON format:

{{
  "answer": "Your answer here or 'I don't know'",
  "confidence_score": 0.0-1.0,
  "source_snippets": {{
    "short snippet from context 1": relevance_score,
    "short snippet from context 2": relevance_score
  }}
}}"""