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


RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert at extracting semantic relationships between entities in educational text.

Given two entities and the sentence containing them, determine:
1. Whether a meaningful, general relationship exists between the entities
2. If yes, express the relationship as a concise verb phrase (2-4 words)

**IMPORTANT: Relationships are DIRECTED**
- The relationship must flow from Entity 1 to Entity 2
- The sentence "Entity 1 [relationship] Entity 2" must make semantic sense
- For example: "softmax function" + "normalizes" + "output" = "softmax function normalizes output" ✓
- If the relationship only makes sense in reverse, output "NO_RELATIONSHIP"

**Guidelines:**
- Only extract direct, general relationships (e.g., "normalizes", "is part of", "trains", "produces")
- Use concise verb phrases that describe the relationship type
- Return "NO_RELATIONSHIP" if:
  - The entities merely co-occur without a clear semantic connection
  - The relationship is only context-specific or instance-based, not a general property
- Focus on relationships that would be true in general, not just in this specific example


**Examples:**
Sentence: "The softmax function normalizes the network output into a probability distribution."
Entity 1: softmax function
Entity 2: network output
Response:
{{
  "reasoning": "The softmax function performs a specific action (normalization) on the network output. This is a general property of the softmax function. Direction check: 'softmax function normalizes network output' ✓",
  "relationship": "normalizes"
}}

Sentence: "Backpropagation is an algorithm for training neural networks."
Entity 1: Backpropagation
Entity 2: neural networks
Response:
{{
  "reasoning": "Backpropagation is explicitly described as having the purpose of training neural networks. This is a general, definitional relationship. Direction check: 'Backpropagation trains neural networks' ✓",
  "relationship": "trains"
}}

Sentence: "Here the derivative is a diagonal matrix."
Entity 1: derivative
Entity 2: diagonal matrix
Response:
{{
  "reasoning": "This describes a property of the derivative in this specific case only ('here'). Derivatives are not always diagonal matrices, so this is context-specific.",
  "relationship": "NO_RELATIONSHIP"
}}

Sentence: "The gradient descent optimizer and the learning rate both affect convergence speed."
Entity 1: gradient descent optimizer
Entity 2: learning rate
Response:
{{
  "reasoning": "Both entities affect the same thing (convergence speed) but there's no direct relationship between them - they're just mentioned together. 'Gradient descent optimizer learning rate' doesn't form a meaningful relationship.",
  "relationship": "NO_RELATIONSHIP"
}}

**First provide your reasoning for your choice, then output your final answer in the following JSON format:**
{{
  "reasoning": "your explanation here",
  "relationship": "verb phrase or NO_RELATIONSHIP"
}}


**Your Context:** {context}

**Your Sentence:** {sentence}

**Entity 1:** {entity1}
**Entity 2:** {entity2}
```"""

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