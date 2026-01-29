import json
import logging
from typing import List, Dict, Optional
from mampfsearch.utils.config import get_llm_client, get_graph_storage


# TODO: Improve that the about_entity actually reflects what the classification is about and not what the segment is about.

# Setup logging based on config (inherits settings)
logger = logging.getLogger("mampfsearch.central_entity")


class SegmentContext:
    def __init__(self, text: str, classification: str, about_entity: str):
        self.text = text
        self.classification = classification
        self.about_entity = about_entity


def get_segment_chain_starts(tx):
    """
    Finds all segments that do not have an incoming IS_SUCCESSOR relationship.
    These are considered the start of a lecture/document chain.
    """
    query = """
    MATCH (s:Segment)
    WHERE NOT ()-[:IS_SUCCESSOR]->(s)
    RETURN elementId(s) as id
    """
    result = tx.run(query)
    return [record["id"] for record in result]


def get_segment_data(tx, segment_id):
    """
    Fetches text, classification (from labels), and linked entities for a specific segment.
    """
    query = """
    MATCH (s:Segment)
    WHERE elementId(s) = $id
    OPTIONAL MATCH (s)-[:MENTIONS_ENTITY]->(e:Entity)
    RETURN s.text as text, 
           labels(s) as labels, 
           collect(e.name) as entities
    """
    result = tx.run(query, id=segment_id).single()
    if result:
        labels = result["labels"]
        # Extract classification label (starts with B- or I- based on classify-all-labels logic)
        classification = "Unknown"
        for label in labels:
            if label.startswith("B-") or label.startswith("I-"):
                classification = label
                break

        return {
            "text": result["text"] or "",
            "classification": classification,
            "entities": result["entities"],
        }
    return None


def get_next_segment_id(tx, current_id):
    """
    Finds the next segment in the chain.
    """
    query = """
    MATCH (s:Segment)-[:IS_SUCCESSOR]->(next:Segment)
    WHERE elementId(s) = $id
    RETURN elementId(next) as id
    """
    result = tx.run(query, id=current_id).single()
    return result["id"] if result else None


def update_segment_node(tx, segment_id, summary, about_entity):
    """
    Updates the segment with the generated summary and about_entity.
    """
    query = """
    MATCH (s:Segment)
    WHERE elementId(s) = $id
    SET s.summary = $summary,
        s.about_entity = $about_entity
    """
    tx.run(query, id=segment_id, summary=summary, about_entity=about_entity)


def generate_analysis(
    llm_client, current_data: Dict, history: List[SegmentContext]
) -> Dict:
    """
    Calls the LLM to determine the central entity and summary.
    """

    # Construct Context String
    context_str = ""
    if not history:
        context_str = "No previous segments (Start of document)."
    else:
        for i, ctx in enumerate(history):
            # i=0 is oldest (prev-2), i=1 is prev-1
            dist = len(history) - i
            context_str += f"""
--- PREVIOUS SEGMENT (Distance: {dist}) ---
Classification: {ctx.classification}
Central Entity (About): {ctx.about_entity}
Text snippet: {ctx.text[:200]}...
"""

    prompt = f"""
You are an expert mathematician analyzing a sequence of text segments from a lecture.
Your goal is to determine the "about_entity" (the central mathematical concept) of the CURRENT segment and write a short summary.

CONTEXT INFORMATION:
{context_str}

--- CURRENT SEGMENT ---
Entities mentioned in text: {", ".join(current_data["entities"])}
Text:
{current_data["text"]}

INSTRUCTIONS:
1. Determine the 'about_entity'. 
   - If this segment continues a proof or explanation from the previous segments, the 'about_entity' is likely the same as the previous one.
   - If it introduces a new definition or theorem, select the most relevant entity from the "Entities mentioned" list or the text.
2. Write a short 'summary'.
   - Examples: "Continuation of the proof for the Spectral Theorem", "Definition of Vector Space", "Remark on the previous lemma".
3. Return ONLY a JSON object with keys "about_entity" and "summary".
"""

    try:
        response = llm_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content
        # Clean potential markdown code blocks
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re

            # This regex looks for backslashes that are NOT followed by valid JSON escape chars (", \, /, b, f, n, r, t, u)
            # and doubles them.
            fixed_content = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", content)
            return json.loads(fixed_content)

    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        # Fallback
        return {"about_entity": "Unknown", "summary": "Error generating summary"}


def main():
    storage = get_graph_storage()
    llm = get_llm_client()

    logger.info("Starting Central Entity Extraction...")

    with storage.driver.session() as session:
        # 1. Find all start nodes (heads of chains)
        start_ids = session.execute_read(get_segment_chain_starts)
        logger.info(f"Found {len(start_ids)} segment chains (lectures/documents).")

        for chain_idx, start_id in enumerate(start_ids):
            logger.info(f"Processing chain {chain_idx + 1}/{len(start_ids)}...")

            # Sliding window for context: stores max 2 previous SegmentContext objects
            history: List[SegmentContext] = []

            current_id = start_id

            while current_id:
                # A. Fetch Data
                data = session.execute_read(get_segment_data, current_id)
                if not data:
                    logger.warning(
                        f"Could not fetch data for segment {current_id}. Stopping chain."
                    )
                    break

                # B. LLM Analysis
                analysis = generate_analysis(llm, data, history)

                about_entity = analysis.get("about_entity", "Unknown")
                summary = analysis.get("summary", "")

                # C. Update Graph
                session.execute_write(
                    update_segment_node, current_id, summary, about_entity
                )

                # D. Update History Window
                new_ctx = SegmentContext(
                    text=data["text"],
                    classification=data["classification"],
                    about_entity=about_entity,
                )
                history.append(new_ctx)
                if len(history) > 2:
                    history.pop(0)  # Remove oldest

                # E. Move to next
                current_id = session.execute_read(get_next_segment_id, current_id)

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
