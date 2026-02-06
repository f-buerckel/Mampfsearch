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


RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert at extracting semantic relationships between entities in educational/math text.

Task:
Given two entities and the text containing them, decide if there is a meaningful, general relationship that holds in a context-independent way. If yes, choose exactly one label from the ALLOWED_RELATIONS below. If no suitable label applies or the direction is wrong, return "NO_RELATIONSHIP".

IMPORTANT: Relationships are DIRECTED
- The relationship must flow from Entity 1 to Entity 2.
- The phrase "Entity 1 [relationship] Entity 2" must be a coherent, correct statement.
- If the relationship only makes sense in reverse, output "NO_RELATIONSHIP".

ALLOWED_RELATIONS (use exactly one of these labels or "NO_RELATIONSHIP"):
- is_a — Taxonomic hierarchy (A is a type of B). Example: "Graph is_a Data Structure"
- part_of — Substructure or containment (A is a component of B). Example: "Vertex part_of Graph"
- instance_of — Specific example (A is an instance of B). Example: "Binary Tree instance_of Tree"
- generalizes — A is more general than B (B is a special case of A). Example: "Function generalizes Linear Function"
- specializes — A is more specific than B (A is a subtype/special case of B). Example: "Linear Function specializes Function"
- defined_by — A is defined/formalized by B (rule, limit, axioms, formula). Example: "Derivative defined_by Limit"
- depends_on — A logically/computationally depends on B. Example: "Integral depends_on Integrand"
- derived_from — A is obtained/computed from B. Example: "Gradient derived_from Loss"
- represented_by — A has representation B (symbolic/graphical/data structure). Example: "Matrix represented_by 2D Array"
- equivalent_to — A is logically equivalent to B (bidirectional). Example: "Undirected Graph equivalent_to Symmetric Adjacency Matrix"
- inverses — A is the inverse of B (mutual inverse). Example: "EncryptionAlgorithm inverses DecryptionAlgorithm"
- implies — A logically implies B. Example: "Finite set implies Countable set"
- contradicts — A is incompatible with B. Example: "P ≠ NP contradicts P = NP"
- proved_by — A (claim/theorem) is proved by B (proof/method). Example: "Theorem proved_by Proof"
- used_in — A (tool/result) is used in B (larger result/method). Example: "Lemma used_in Theorem"
- applies_to — A (method/rule) applies to B (object/domain). Example: "Algorithm applies_to Graph"
- solves — A (method/algorithm) solves B (problem/task). Example: "Dijkstra's algorithm solves Shortest Path"
- computes — A (function/algorithm) computes B (quantity/output). Example: "Softmax computes Probabilities"
- requires — A (method/algorithm) requires B (input/hyperparameter/assumption). Example: "Gradient Descent requires Learning Rate"
- complexity_of — A (e.g., O(n log n)) is the complexity of B (algorithm/problem). Example: "O(n log n) complexity_of MergeSort"

Guidelines:
- Choose a relation only if it is general and context-independent.
- Prefer the most specific applicable label.
- Mere co-occurrence or instance-specific/temporary states → "NO_RELATIONSHIP".
- Symmetric relations (equivalent_to, contradicts, inverses) still require correct direction: "Entity 1 [relation] Entity 2" must read sensibly.
- If uncertain or no allowed label fits, return "NO_RELATIONSHIP".

Output format:
First provide brief reasoning, then output JSON:
{{
  "reasoning": "your explanation here",
  "relationship": "<one of the labels above or NO_RELATIONSHIP>"
}}

Examples:

Sentence: "Backpropagation is an algorithm for training neural networks."
Entity 1: Backpropagation
Entity 2: neural networks
Response:
{{
  "reasoning": "Backpropagation is a method applicable to neural networks in general.",
  "relationship": "applies_to"
}}

Sentence: "O(n log n) is the time complexity of MergeSort."
Entity 1: O(n log n)
Entity 2: MergeSort
Response:
{{
  "reasoning": "The expression denotes the complexity class of MergeSort.",
  "relationship": "complexity_of"
}}

Sentence: "The softmax function outputs probabilities."
Entity 1: softmax function
Entity 2: probabilities
Response:
{{
  "reasoning": "Softmax generally computes probabilities from logits.",
  "relationship": "computes"
}}

Sentence: "Here the derivative is a diagonal matrix."
Entity 1: derivative
Entity 2: diagonal matrix
Response:
{{
  "reasoning": "This is a context-specific, temporary statement, not a general relation.",
  "relationship": "NO_RELATIONSHIP"
}}

Now decide for the following:

Your Text: {context}

Entity 1: {entity1}
Entity 2: {entity2}
"""

NER_VALIDATION_PROMPT = """You are an expert at validating named entities in mathematical and educational text.

Your task is to determine if an extracted entity is a valid mathematical concept, technique, or object that should be included in a knowledge graph.

**Valid entities are:**
- Mathematical concepts with established names that make sense in a general context (e.g., "vector", "matrix", "gradient descent")
- Named theorems, laws, or principles (e.g., "Theorem of Pythagoras", "Central Limit Theorem", "Chain Rule")
- Mathematical operations or functions with standard names (e.g., "dot product", "cross product", "softmax function")
- Algorithms or methods with recognized names (e.g., "backpropagation", "Newton's method", "LU decomposition")
- Mathematical structures or spaces (e.g., "Hilbert space", "vector space", "probability distribution")
- Entities that represent general mathematical concepts, not instance-specific references

**Invalid entities are:**
- Generic variables or placeholders (e.g., "x", "y", "k", "n", "i")
- Mathematical operators or symbols (e.g., "<", ">", "=", "+", "∈")
- Numbered references without descriptive names (e.g., "Theorem 1.5", "Equation 3.2", "Figure 2")
- Common adjectives or descriptors alone (e.g., "large", "small", "positive")
- Generic words that are not mathematical terms (e.g., "example", "case", "result")
- Pronouns or demonstratives (e.g., "this", "that", "it")
- Context-specific phrases that only make sense in a particular lecture/document (e.g., "our method", "the previous example")
- Entities that are too vague or generic to be meaningful outside their immediate context

**Key principle:** The entity should be recognizable and meaningful to someone in the mathematical field without needing the specific context where it appeared.

**Positive Examples:**
- Vector ✓ (general mathematical concept)
- Matrix ✓ (general mathematical concept)
- Theorem of Pythagoras ✓ (named theorem)
- dot product ✓ (standard operation)
- gradient descent ✓ (established algorithm)
- eigenvalue ✓ (well-defined concept)
- linear regression ✓ (recognized method)
- Fourier transform ✓ (named transformation)
- neural network ✓ (general concept)

**Negative Examples:**
- x ✗ (generic variable)
- k ✗ (generic variable)
- < ✗ (operator symbol)
- Theorem 1.5 ✗ (numbered reference without name)
- Equation 2 ✗ (numbered reference)
- positive ✗ (generic adjective)
- this ✗ (pronoun)
- our approach ✗ (context-specific, not a general concept)
- the result ✗ (too vague, context-dependent)

**Entity to validate:** {entity}

**Respond with only "yes" or "no".**"""

RELATIONSHIP_VALIDATION_PROMPT = """You are an expert at validating semantic relationships between mathematical entities.

Your task is to determine if an extracted relationship represents a general, context-independent connection between two mathematical concepts that would be meaningful in a knowledge graph.

**IMPORTANT: Relationships are DIRECTED**
- The relationship flows from Entity 1 to Entity 2
- Verify that the sentence "Entity 1 [relationship] Entity 2" makes semantic sense
- If the relationship doesn't work in this direction, respond with "no"

**Valid relationships are:**
- General properties or actions that are inherently true (e.g., "softmax function normalizes output")
- Definitional relationships (e.g., "eigenvalue characterizes linear transformation")
- Standard mathematical connections (e.g., "gradient descent minimizes loss function")
- Relationships that describe what entities do, produce, or are part of in general (e.g., "backpropagation trains neural networks")
- Connections that would be recognized by mathematicians without needing specific context
- **Must form a coherent sentence: Entity 1 + relationship + Entity 2**

**Invalid relationships are:**
- Instance-specific or temporary states (e.g., "derivative is diagonal matrix" - only true in one specific case)
- Context-dependent relationships that rely on a particular example (e.g., "x equals 5" - only in that example)
- Relationships that use demonstratives or context-specific references (e.g., "this result follows from that theorem")
- Mere co-occurrence without semantic connection (e.g., "gradient descent and learning rate" mentioned together)
- Relationships that describe what happens in a specific scenario rather than general properties (e.g., "here the matrix is invertible")
- Time-bound or conditional relationships (e.g., "now the error decreases", "if positive then increasing")
- **Relationships that don't make sense when read as "Entity 1 [relationship] Entity 2"**

**Key principles:**
1. Ask yourself: "Would this relationship be true and meaningful in a general mathematical encyclopedia or textbook definition, not just in this specific lecture or example?"
2. **Check directionality: Does "Entity 1 [relationship] Entity 2" form a coherent correct statement?**

**Positive Examples:**
- "softmax function" --[normalizes]--> "output" ✓ (general property, correct direction: "softmax function normalizes output")
- "backpropagation" --[trains]--> "neural networks" ✓ (definitional purpose, correct direction: "backpropagation trains neural networks")
- "eigenvalue" --[characterizes]--> "linear transformation" ✓ (standard connection, correct direction: "eigenvalue characterizes linear transformation")
- "gradient" --[indicates direction of]--> "steepest ascent" ✓ (mathematical definition, correct direction: "gradient indicates direction of steepest ascent")
- "determinant" --[measures]--> "matrix invertibility" ✓ (general property, correct direction: "determinant measures matrix invertibility")

**Negative Examples:**
- "derivative" --[is]--> "diagonal matrix" ✗ (only in specific case)
- "x" --[equals]--> "5" ✗ (instance-specific value)
- "result" --[follows from]--> "previous theorem" ✗ (context-dependent reference)
- "here" --[produces]--> "positive value" ✗ (temporary state with demonstrative)
- "matrix" --[is]--> "invertible" ✗ (not true for all matrices, context-specific)
- "gradient descent" --[uses]--> "learning rate" ✗ (co-occurrence, not semantic relationship)
- "neural networks" --[backpropagation]--> "trained" ✗ (wrong direction, doesn't make grammatical sense)

**Context:** {context}

**Sentence:** {sentence}

**Entity 1:** {entity1}

**Entity 2:** {entity2}

**Proposed Relationship:** {relationship}

**Questions to consider:**
1. Would this relationship be true and useful in a general mathematical knowledge base, or is it only relevant to this specific example/context?
2. **Does the phrase "{entity1} {relationship} {entity2}" make semantic sense?**

**Respond with only "yes" or "no".**"""

CREATE_FACTUAL_QUESTION_PROMPT = """
You are an expert mathematician and dataset curator creating a quiz database from university lecture transcripts. 

Your goal is to extract ONE precise, factual question about a specific [Entity] based ONLY on the provided text segment.

The generated question must be **standalone**: it should make sense to a reader who has NOT seen the transcript, while still being answerable from the given segment.

### RULES FOR "VALID" SEGMENTS:
1. The segment must contain a DEFINITION, a PROPERTY, a THEOREM, or a CONDITION regarding the [Entity].
2. The information must be self-contained in the text (or clearly implied by immediate context).
3. IGNORE conversational filler (e.g., "So, um, basically...", "Next slide please", "Is that clear?").
4. IGNORE meta-commentary (e.g., "We will discuss this later", "This is important for the exam").

### RULES FOR THE QUESTION (CRITICAL):
1. The question should not depend on the context of the given segment beyond what is explicitly stated in the question itself.
2. Do NOT use deictic or context-dependent references: avoid "this", "that", "here", "it", "the above", "the following", "we", "our", "in this lecture", "in the slide", etc.
3. Do NOT ask counterfactual or "replace/if we changed X" questions (e.g., "What would happen if...", "What series would you get if...", "If instead... then...") even if the segment mentions an example like that. Those are NOT stable factual questions.
  - Exception: general mathematical conditionals that are themselves the stated fact are allowed (e.g., "If $f$ is continuous on [a,b], then ...").
4. The question must be specific and unambiguous: include all necessary math objects in the question statement (e.g., name the function/series/space explicitly), not by pointing.
5. The answer must be extractable verbatim (or as a minimal exact paraphrase) from the segment.

### INSTRUCTIONS:
1. Read the Input Entity and the Transcript Segment.
2. Determine if the segment contains a hard mathematical fact about the Entity.
3. If YES: Generate a specific standalone question and the answer found in the text.
4. If the segment is mostly an informal example, conversational, meta, or would only yield a counterfactual/"replace X" style question: Return nulls.

### OUTPUT FORMAT:
You must output a single valid JSON object with the following structure:
{{
  "reasoning": "Brief explanation of why this segment is valid or invalid",
  "contains_fact": boolean,
  "question": "The generated question string OR null",
  "answer": "The answer derived strictly from text OR null"
}}

INPUT:
{{ 
  "entity": "{entity}",
  "segment": "{segment}"
}}
"""
CREATE_MULTIPLE_CHOICE_QUESTION_PROMPT = """
You are an expert mathematician and dataset curator creating a quiz database from university lecture transcripts.
Your goal is to create a MULTIPLE-CHOICE question based ONLY on the provided text segment about the given entity.

The generated question must be **standalone**: it should make sense to a reader who has NOT seen the transcript, while still being answerable from the given segment.

### RULES FOR "VALID" SEGMENTS:
1. The segment must contain a DEFINITION, a PROPERTY, a THEOREM, or a CONDITION regarding the [Entity].
2. The information must be self-contained in the text (or clearly implied by immediate context).
3. IGNORE conversational filler (e.g., "So, um, basically...", "Next slide please", "Is that clear?").
4. IGNORE meta-commentary (e.g., "We will discuss this later", "This is important for the exam").

### RULES FOR THE MULTIPLE-CHOICE QUESTION (CRITICAL):
1. The question should not depend on the context of the given segment beyond what is explicitly stated in the question itself.
2. Do NOT use deictic or context-dependent references: avoid "this", "that", "here", "it", "the above", "the following", "we", "our", "in this lecture", "in the slide", etc.
3. Do NOT ask counterfactual or "replace/if we changed X" questions (e.g., "What would happen if...", "What series would you get if...", "If instead... then...") even if the segment mentions an example like that. Those are NOT stable factual questions.
  - Exception: general mathematical conditionals that are themselves the stated fact are allowed (e.g., "If $f$ is continuous on [a,b], then ...").
4. The question must be specific and unambiguous: include all necessary math objects in the question statement (e.g., name the function/series/space explicitly), not by pointing.
5. The correct answer must be extractable verbatim (or as a minimal exact paraphrase) from the segment.
6. Create THREE plausible distractors (wrong answer choices):
   - Distractors must be related to the topic but clearly incorrect.
   - Distractors should be similar in length and complexity to the correct answer.
   - Avoid using "all of the above", "none of the above", or similar options.

### INSTRUCTIONS:
1. Read the Input Entity and the Transcript Segment.
2. Determine if the segment contains a hard mathematical fact about the Entity.
3. If YES: Generate a specific standalone multiple-choice question, the correct answer found in the text, and FOUR plausible distractors.
4. If the segment is mostly an informal example, conversational, meta, or would only yield a counterfactual/"replace X" style question: Return nulls.

### OUTPUT FORMAT:
You must output a single valid JSON object with the following structure:
{{
  "reasoning": "Brief explanation of why this segment is valid or invalid",
  "contains_fact": boolean,
  "question": "The generated multiple-choice question string",
  "answer": "The correct answer derived strictly from text",
  "distractor1": "First plausible wrong answer",
  "distractor2": "Second plausible wrong answer",
  "distractor3": "Third plausible wrong answer",
}}
or NULL if no valid question can be generated.

INPUT:
{{ 
  "entity": "{entity}",
  "segment": "{segment}",
}}
"""


CREATE_SPANNING_QUESTION_PROMPT = """
You are an expert mathematician creating an assessment question for a lecture summary.
Your goal is to test whether a student understood the specific logical flow, motivations, or definitions presented in the provided text about "{entity}" by generating {n_questions} questions.

**Strict Content Boundaries**:
1.  **Source Truth**: You must generate the question and answer based **ONLY** on the information, examples, and logic explicitly present in the "Context".
2.  **No Outside Knowledge**: Do NOT ask for proofs, calculations, or definitions that are far outside the scope of this specific text. (e.g., If the text mentions "Banach spaces exist" but doesn't define the norm properties, do not ask the student to prove the triangle inequality).

**Phrasing Constraints**:
1.  Do NOT refer to "the text provided", "the passage", or "the snippet".
2.  Instead, use natural phrasing like "In the lecture...", "According to the discussion on {entity}...", or "As explained regarding..."
3.  The question must be standalone.

**Mathematical Formatting**:
-   ALWAYS use valid LaTeX for mathematical symbols and equations (e.g., `\\mathbb{{R}}`, `\\epsilon > 0`).
-   Do NOT use Unicode math characters (use `\\in` not `∈`).

Context:
{context}

Return the output in the following JSON format:
{{
  "questions": ["The first question text", "The second question text", "..."],
  "answers": ["Answer to the first question", "Answer to the second question", "..."],
  "explanations": ["Brief explanation for Q1", "Brief explanation for Q2", "..."]
}}
"""


CREATE_MULTI_ENTITY_SPANNING_QUESTION_PROMPT = """
You are an expert mathematician creating assessment questions for a lecture.
Your goal is to test whether a student can reason about "{entity}" *together with other mathematical entities* mentioned in the lecture by generating up to {n_questions} questions.

**Strict Content Boundaries**:
1. **Source Truth**: You must generate the question and answer based **ONLY** on the information, examples, and logic explicitly present in the "Context".
2. **No Outside Knowledge**: Do NOT ask for proofs, calculations, or definitions that are far outside the scope of the provided context.

**Multi-Entity Reasoning Requirement**:
1. Each question MUST involve "{entity}" and if somehow possible at least ONE additional entity from: {other_entities}.
2. The question must require reasoning across entities (e.g., connecting a definition of one entity to a property/usage of another), not mere mention.
3. If possible, create questions that combine multiple concepts mentioned to test deeper understanding and generate questions that require deeper reasoning across the text, not just simple fact recall.


**Context Structure (IMPORTANT)**:
- The context is separated into multiple blocks (e.g., Main Text, Related Entity Descriptions, Definition Context, Co-mention Context).
- These blocks are **separate evidence sources** and may be from different parts of the lecture.
- Do NOT assume the blocks are consecutive in the lecture.
- Do NOT refer to block names ("Main Text", "Definition Context", etc.) in the question.

**Phrasing Constraints**:
1. Do NOT refer to "the text provided", "the passage", or "the snippet".
2. Use natural phrasing like "In the lecture...", "As discussed...", or "According to the discussion on ...".
3. The question must be standalone.

**Mathematical Formatting**:
- ALWAYS use valid LaTeX for mathematical symbols and equations (e.g., `\\mathbb{{R}}`, `\\epsilon > 0`).
- Do NOT use Unicode math characters (use `\\in` not `∈`).

Context:
{context}

Return the output in the following JSON format:
{{
  "questions": ["..."],
  "answers": ["..."],
  "explanations": ["..."]
}}
"""

CREATE_UNSTRUCTURED_QUESTION_PROMPT = """
You are an expert mathematician creating an assessment question for a lecture summary.
Your goal is to test whether a student understood the specific logical flow, motivations, or definitions presented in the provided text about {entity} by generating {n_questions} questions.

**Strict Content Boundaries**:
1.  **Source Truth**: You must generate the question and answer based **ONLY** on the information, examples, and logic explicitly present in the lecture.
2.  **No Outside Knowledge**: Do NOT ask for proofs, calculations, or definitions that are far outside the scope of this specific text. (e.g., If the text mentions "Banach spaces exist" but doesn't define the norm properties, do not ask the student to prove the triangle inequality).

**Phrasing Constraints**:
1.  Do NOT refer to "the text provided", "the passage", or "the snippet".
2.  Instead, use natural phrasing like "In the lecture...", "According to the discussion on {entity}...", or "As explained regarding..."
3.  The question must be standalone.

**Mathematical Formatting**:
-   ALWAYS use valid LaTeX for mathematical symbols and equations (e.g., `\\mathbb{{R}}`, `\\epsilon > 0`).
-   Do NOT use Unicode math characters (use `\\in` not `∈`).

Context:
{context}

Return the output in the following JSON format:
{{
    "questions": ['The first question text', 'The second question text', ... 'Question {n_questions} text'],
    "answers": ['Answer to first question', 'Answer to second question', ... 'Answer to question {n_questions}'],
}}
"""

CREATE_UNSTRUCTURED_QUESTION_PROMPT_NO_ENTITY = """
You are an expert mathematician creating assessment questions for a lecture.
Your goal is to test whether a student understood the specific logical flow, motivations, or definitions presented in the provided lecture excerpt by generating up to {n_questions} questions.
If possible, create questions that combine multiple concepts mentioned to test deeper understanding and generate questions that require deeper reasoning across the text, not just simple fact recall.


**Strict Content Boundaries**:
1.  **Source Truth**: You must generate the questions and answers based **ONLY** on the information, examples, and logic explicitly present in the lecture excerpt.
2.  **No Outside Knowledge**: Do NOT ask for proofs, calculations, or definitions that are far outside the scope of this specific excerpt.

**Phrasing Constraints**:
1.  Do NOT refer to "the text provided", "the passage", or "the snippet".
2.  Instead, use natural phrasing like "In the lecture..." or "As explained...".
3.  The question must be standalone.

**Mathematical Formatting**:
-   ALWAYS use valid LaTeX for mathematical symbols and equations (e.g., `\\mathbb{{R}}`, `\\epsilon > 0`).
-   Do NOT use Unicode math characters (use `\\in` not `∈`).

Context:
{context}

Return the output in the following JSON format:
{{
  "questions": ['The first question text', 'The second question text', ... 'Question x <= {n_questions} text'],
  "answers": ['Answer to first question', 'Answer to second question', ... 'Answer to question {n_questions}'],
}}
"""

EVALUATION_PROMPT = """
### SYSTEM ROLE
You are an expert university pedagogue and NLP evaluator. Your task is to rigorously evaluate the quality of a generated exam question-answer pair based *only* on the provided source context.

### INPUT DATA
- **Source Context:** {context}
- **Generated Question:** {question}
- **Generated Answer:** {answer}

### EVALUATION CRITERIA
Rate each of the following 9 criteria on a scale of 1-5.

#### A. Linguistic Quality
1. **Clarity**: Is the question unambiguous and specific?
   - 1: Vague or confusing; multiple interpretations possible.
   - 5: Crystal clear intent and meaning.
2. **Conciseness**: Is it free from unnecessary verbosity?
   - 1: Extremely verbose or contains redundant modifiers.
   - 5: Concise and to the point.

#### B. Content Alignment (Grounding)
3. **Relevance**: Does it target key information from the text?
   - 1: Asks about trivial/irrelevant details.
   - 5: Targets the core concept of the passage.
4. **Consistency**: Is it factually aligned with the passage?
   - 1: Contradicts the text or hallucinates facts.
   - 5: Perfectly consistent with the source.
5. **Answerability**: Can the answer be found in the given context and only very basic external knowledge?
   - 1: Impossible to answer given only this context chunk.
   - 5: Answer is explicitly and clearly in the text.

6. **Answer Consistency**: Does the provided answer actually answer the question asked?
  - 1: The answer does not match the question (wrong target, incomplete, or off-topic).
  - 5: The answer perfectly addresses the specific question.

#### C. Pedagogical Value
7. **Educational Complexity**: What level of cognitive effort is required (Bloom's Taxonomy) and how much information is needed? Does it require more than a single fact or concept?
  - 1: Simple Recall (e.g., "What is X?").
  - 2: Basic Understanding (e.g., "State the definition/meaning of X." / "Identify a stated property of X.").
  - 3: Application/Inference (e.g., "Why does X happen?" / "Use the stated rule to determine ...").
  - 4: Analysis/Integration (e.g., "How does X relate to Y according to the text?" / "Explain a consequence of the definition.").
  - 5: Synthesis/Evaluation (e.g., "Compare X and Y..." / "Argue which approach is preferable and why.").

8. **Independence (Answer Leakage)**: Does the question leak the answer?
   - 1: The question gives away the answer (e.g., "Since X is Y, why...?").
   - 5: The question stands alone without hinting at the solution.
#### D. Holistic Assessment
9. **Overall Quality**: A holistic score considering all factors, with a strong emphasis on educational objectives.
  - Educational value is crucial: prioritize questions that test understanding and the combination/integration of concepts (i.e., higher Educational Complexity).
  - Penalize questions that are merely simple recall, even if they are clear and factually consistent.
  - If a question fails on critical issues (e.g., hallucination/ungrounded content, or answer leakage), this score should be low regardless of linguistic features.

### OUTPUT FORMAT
Return valid JSON only. Structure:
{{
  "clarity": {{ "reasoning": "...", "score": <int> }},
  "conciseness": {{ "reasoning": "...", "score": <int> }},
  "relevance": {{ "reasoning": "...", "score": <int> }},
  "consistency": {{ "reasoning": "...", "score": <int> }},
  "answerability": {{ "reasoning": "...", "score": <int> }},
  "answer_consistency": {{ "reasoning": "...", "score": <int> }},
  "educational_complexity": {{ "reasoning": "...", "score": <int> }},
  "independence": {{ "reasoning": "...", "score": <int> }},
  "overall_review": {{ "reasoning": "...", "score": <int> }}
}}
"""

FULL_PIPELINE_PROMPT = """You are an expert mathematical NER + relation + property extractor.

Input JSON:
{{ "text": {context} }}

Tasks:
1. Extract entities (ALGORITHM | FUNCTION | THEOREM_RULE | OPERATOR | DISTRIBUTION | CONCEPT | CONSTANT).
2. Extract directed relations between entities using this controlled verb set:
   is_a | uses | minimizes | maximizes | optimizes_for | applied_to | implies | equals | maps_to | part_of | has_hyperparameter | used_to_compute | influences | converges_to | involves | allows | defines
3. Extract explicit properties (numbers or strings) stated about entities (value, formula, example, etc.).
Output STRICT JSON (no comments):
{{
  "text": "...",
  "context": "...",
  "entities": [
    {{ "id": "E#", "text": "...", "label": "...", "canonical": "..." }}
  ],
  "relations": [
    {{ "subject": "E#", "relation": "...", "object": "E#" }}
  ],
  "properties": [
    {{ "entity": "E#", "property": "...", "value": <number|string>, "value_type": "number|string", "unit": null|"...", "approximate": true|false }}
  ]
}}

Rules:
- Exclude variables (x, k, i), set symbols (R, N), local placeholders, numbered theorems without names.
- Include constants (pi, e), standard operators (argmax, dot product), named functions (softmax, logistic sigmoid), losses, optimizers, hyperparameters.
- Canonical form should be normalized (e.g., "logistic sigmoid function" -> "logistic sigmoid").
- Properties examples: has_value, defining_formula, example_value, base, exponent, tolerance.
- Mark approximate if sentence uses ≈, about, roughly.
- defining_formula values should be verbatim string (LaTeX or plain).
- Relationships are DIRECTED: subject [relation] object must make sense.
- Relationships must make general sense, not just context-specific.
- If nothing extracted return empty arrays.

Examples:

EX1:
Input:
{{"text":"LDA is a linear classifier using Gaussian assumptions.","context":"Generative models overview."}}
Output:
{{
  "text":"LDA is a linear classifier using Gaussian assumptions.",
  "context":"Generative models overview.",
  "entities":[
    {{"id":"E1","text":"LDA","label":"ALGORITHM","canonical":"LDA"}},
    {{"id":"E2","text":"linear classifier","label":"CONCEPT","canonical":"linear classifier"}},
    {{"id":"E3","text":"Gaussian","label":"DISTRIBUTION","canonical":"Gaussian"}}
  ],
  "relations":[
    {{"subject":"E1","relation":"is_a","object":"E2"}},
    {{"subject":"E1","relation":"uses","object":"E3"}}
  ],
  "properties":[]
}}

EX2 (formula property):
Input:
{{"text":"The dot product has defining formula sum_{{i=1}}^n u_i v_i.","context":"Vector operations."}}
Output:
{{
  "text":"The dot product has defining formula sum_{{i=1}}^n u_i v_i.",
  "context":"Vector operations.",
  "entities":[
    {{"id":"E1","text":"dot product","label":"OPERATOR","canonical":"dot product"}}
  ],
  "relations":[],
  "properties":[
    {{"entity":"E1","property":"defining_formula","value":"sum_{{i=1}}^n u_i v_i","value_type":"string","unit":null,"approximate":false}}
  ]
}}

EX3 (function + mapping + formula):
Input:
{{"text":"The logistic sigmoid function maps inputs to probabilities and its defining formula is 1/(1+e^{{-z}}).","context":"Binary classification."}}
Output:
{{
  "text":"The logistic sigmoid function maps inputs to probabilities and its defining formula is 1/(1+e^{{-z}}).",
  "context":"Binary classification.",
  "entities":[
    {{"id":"E1","text":"logistic sigmoid function","label":"FUNCTION","canonical":"logistic sigmoid"}},
    {{"id":"E2","text":"probabilities","label":"CONCEPT","canonical":"probabilities"}}
  ],
  "relations":[
    {{"subject":"E1","relation":"maps_to","object":"E2"}}
  ],
  "properties":[
    {{"entity":"E1","property":"defining_formula","value":"1/(1+e^{{-z}})","value_type":"string","unit":null,"approximate":false}}
  ]
}}

EX4 (constant value approximate):
Input:
{{"text":"Let pi be ≈ 3.14159 today.","context":""}}
Output:
{{
  "text":"Let pi be ≈ 3.14159 today.",
  "context":"",
  "entities":[
    {{"id":"E1","text":"pi","label":"CONSTANT","canonical":"pi"}}
  ],
  "relations":[],
  "properties":[
    {{"entity":"E1","property":"has_value","value":3.14159,"value_type":"number","unit":null,"approximate":true}}
  ]
}}

EX5 (example value):
Input:
{{"text":"Real numbers include examples like 3.12521.","context":"Intro to reals."}}
Output:
{{
  "text":"Real numbers include examples like 3.12521.",
  "context":"Intro to reals.",
  "entities":[
    {{"id":"E1","text":"Real numbers","label":"CONCEPT","canonical":"real numbers"}}
  ],
  "relations":[],
  "properties":[
    {{"entity":"E1","property":"example_value","value":3.12521,"value_type":"number","unit":null,"approximate":false}}
  ]
}}

EX6 (optimizer hyperparameter numeric):
Input:
{{"text":"Use Adam with learning rate 1e-3 and beta1 0.9.","context":"Training setup."}}
Output:
{{
  "text":"Use Adam with learning rate 1e-3 and beta1 0.9.",
  "context":"Training setup.",
  "entities":[
    {{"id":"E1","text":"Adam","label":"ALGORITHM","canonical":"Adam"}},
    {{"id":"E2","text":"learning rate","label":"CONCEPT","canonical":"learning rate"}},
    {{"id":"E3","text":"beta1","label":"CONCEPT","canonical":"beta1"}}
  ],
  "relations":[
    {{"subject":"E1","relation":"has_hyperparameter","object":"E2"}},
    {{"subject":"E1","relation":"has_hyperparameter","object":"E3"}}
  ],
  "properties":[
    {{"entity":"E2","property":"has_value","value":0.001,"value_type":"number","unit":null,"approximate":false}},
    {{"entity":"E3","property":"has_value","value":0.9,"value_type":"number","unit":null,"approximate":false}}
  ]
}}

EX7 (theorem applied_to concept):
Input:
{{"text":"We apply the Cauchy–Schwarz inequality to the inner product.","context":"Bounding norms."}}
Output:
{{
  "text":"We apply the Cauchy–Schwarz inequality to the inner product.",
  "context":"Bounding norms.",
  "entities":[
    {{"id":"E1","text":"Cauchy–Schwarz inequality","label":"THEOREM_RULE","canonical":"Cauchy–Schwarz inequality"}},
    {{"id":"E2","text":"inner product","label":"OPERATOR","canonical":"inner product"}}
  ],
  "relations":[
    {{"subject":"E1","relation":"applied_to","object":"E2"}}
  ],
  "properties":[]
}}

EX8 (loss minimized + formula):
Input:
{{"text":"Cross-entropy loss minimized by gradient descent has defining formula -\\sum y_i log p_i.","context":"Classification."}}
Output:
{{
  "text":"Cross-entropy loss minimized by gradient descent has defining formula -\\sum y_i log p_i.",
  "context":"Classification.",
  "entities":[
    {{"id":"E1","text":"Cross-entropy loss","label":"FUNCTION","canonical":"cross-entropy loss"}},
    {{"id":"E2","text":"gradient descent","label":"ALGORITHM","canonical":"gradient descent"}}
  ],
  "relations":[
    {{"subject":"E2","relation":"minimizes","object":"E1"}}
  ],
  "properties":[
    {{"entity":"E1","property":"defining_formula","value":"-\\sum y_i log p_i","value_type":"string","unit":null,"approximate":false}}
  ]
}}

EX9 (argmax operator definition):
Input:
{{"text":"Argmax defines the index of the largest value.","context":"Decision rule."}}
Output:
{{
  "text":"Argmax defines the index of the largest value.",
  "context":"Decision rule.",
  "entities":[
    {{"id":"E1","text":"Argmax","label":"OPERATOR","canonical":"argmax"}},
    {{"id":"E2","text":"index","label":"CONCEPT","canonical":"index"}},
    {{"id":"E3","text":"value","label":"CONCEPT","canonical":"value"}}
  ],
  "relations":[
    {{"subject":"E1","relation":"defines","object":"E2"}}
  ],
  "properties":[]
}}

If extraction uncertain -> empty arrays. Output must be valid JSON only.
Return JSON now.
"""

CREATE_2HOP_QUESTION_PROMPT = """
You are an expert mathematician and dataset curator. Your task is to generate a meaningful **2-Hop Question** based on a logical chain from a Knowledge Graph.

The goal is to test if a student understands the connection between a **Target Concept** and a specific **Reference Entity** by identifying the **Bridge Concept** that links them.

### INPUT DATA STRUCTURE:
You will receive a logical chain:
1. **Target (Answer)**: The concept the student must identify.
2. **Bridge**: The intermediate concept that links the Target to the Reference.
3. **Reference**: The concept, definition, or example used to frame the question.
4. **Documents**: Text segments verifying these links.

### THE LOGIC (2-HOPS):
* **Hop 1:** Target is related to Bridge (e.g., "A" *relies on* "B").
* **Hop 2:** Bridge is related to Reference (e.g., "B" *is a theorem about* "C").

### RULES FOR THE QUESTION:
1.  **Synthesize, Don't List:** Do not ask "What is related to A and B?". Instead, ask: "What [Target] [Relationship 1] the [Bridge] which [Relationship 2] [Reference]?"
2.  **Hide the Bridge (Optional):** To increase difficulty, you can describe the Bridge by its properties rather than naming it, forcing the student to infer the connection.
3.  **Handle Context:** If the Reference is a variable (e.g., "$x_n$") defined only in `doc_b_preceding`, you MUST include that definition in the question stem.
4.  **Directionality:** The question must clearly ask for the **Target**.

### ONE-SHOT EXAMPLE:

**INPUT:**
{{
  "target_entity": "Extreme Value Theorem",
  "rel_1": "uses in proof",
  "bridge_entity": "Bolzano-Weierstrass Theorem",
  "rel_2": "states property of",
  "reference_entity": "Bounded Sequences",
  "doc_a": "The Extreme Value Theorem (EVT) is a central result for continuous functions... The proof relies on the Bolzano-Weierstrass Theorem.",
  "doc_b": "The Bolzano-Weierstrass Theorem states that every bounded sequence of real numbers has at least one convergent subsequence.",
  "doc_b_preceding": "Recall the definition of a sequence definition."
}}

**OUTPUT:**
{{
  "reasoning": "The chain is valid. Doc A links EVT (Target) to Bolzano-Weierstrass (Bridge). Doc B links Bolzano-Weierstrass to 'Bounded Sequences' (Reference). The logic holds: EVT -> uses -> BW -> is about -> Bounded Sequences. I will construct a question asking for EVT, referencing the property of bounded sequences provided by BW.",
  "contains_valid_chain": true,
  "question": "Which theorem relies on the Bolzano-Weierstrass Theorem to guarantee its result, utilizing the fact that every **bounded sequence** has a convergent subsequence?",
  "answer": "Extreme Value Theorem"
}}

### INSTRUCTIONS:
1. Analyze the Chain and Documents. Verify the text supports the relationships.
2. If `reference_entity` relies on `doc_b_preceding` for its definition, fuse that text into the question.
3. Generate a JSON response following the example above.

### ACTUAL INPUT:
{{
  "target_entity": "{target_entity}",
  "rel_1": "{rel_1}",
  "bridge_entity": "{bridge_entity}",
  "rel_2": "{rel_2}",
  "reference_entity": "{reference_entity}",
  "doc_a": "{doc_a}",
  "doc_b": "{doc_b}",
  "doc_b_preceding": "{doc_b_preceding}"
}}
"""


QUESTION_EVALUATION_PROMPT = """
### SYSTEM ROLE
You are a strict Mathematical Pedagogy Evaluator. Your job is to audit a Question-Answer (QA) pair generated from a University-level mathematics lecture transcript. The QA pair was generated based on specific entities found in a Knowledge Graph.

### INPUT DATA
<Context>
{context}
</Context>

<Target_Entity>
{entity_name}
</Target_Entity>

<Generated_Question>
{question}
</Generated_Question>

<Generated_Answer>
{answer}
</Generated_Answer>

### EVALUATION CRITERIA
Evaluate the QA pair on the following 4 metrics.

**1. Faithfulness (1-5)**
- Does the Answer rely *only* on the information in the <Context>?
- Penalize heavily if the Answer uses outside knowledge not present in the lecture (even if mathematically correct), unless it is common knowledge required to understand the language.
- Score 1 if the answer contradicts the context.

**2. Entity Alignment (1-5)**
- Is the <Target_Entity> central to the question?
- A score of 5 means the question tests deep understanding of the {entity_name}.
- A score of 1 means the entity is mentioned only in passing or not at all.

**3. Mathematical Correctness (1-5)**
- Is the reasoning in the Answer logically sound?
- Check for causality errors (e.g., confusing "necessary" vs "sufficient" conditions).
- Check for notation errors if LaTeX is used.

**4. Pedagogical Utility (1-5)**
- Is this a useful question for a student studying for an exam?
- Score 1 for "Trivial text matching" (e.g., "What is the definition of X?" when the text says "X is defined as...").
- Score 5 for "Synthesis/Application" (e.g., "Given the conditions in the lecture, why does property X hold?").

### EVALUATION STEPS (Chain of Thought)
1. Read the Context and identify the key mathematical claims regarding {entity_name}.
2. Analyze the Question: Does it target those claims?
3. Analyze the Answer: Is it supported by the text? Is the logic valid?
4. Determine if the question challenges the student or just requires copy-pasting.

### OUTPUT FORMAT
Provide your response in the following JSON format only:

{{
  "reasoning_steps": "Detailed analysis of how you reached your conclusion, citing specific phrases from the context.",
  "faithfulness": <int>,
  "entity_alignment": <int>,
  "math_correctness": <int>,
  "pedagogical_utility": <int>
  "verdict": "<'KEEP' or 'DISCARD'>",
}}
"""
