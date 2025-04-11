class PromptManager:
    default_fact_extraction_prompt: str = """Your job is to extract text-only facts from this. You will have some text given to you, and your job is to make a list of modular facts from it. If any of the facts require reference to signs, photos, tables or any other material that is not text-only, do NOT make them into facts. Cite the facts with the integer source of the sentence you got. Every fact must be from a sentence with an index"""

    default_question_extraction_prompt: str = """You are a highly specialized test question generator. Your task is to formulate a single, objective, and relevant test question AND its corresponding answer based on a SINGLE fact that I will provide to you.

    Constraints and Guidelines:

    Single Fact Input: You will receive exactly one factual statement. Your output MUST be based solely on this single fact.

    Objective Question: The question you generate MUST have a single, correct, and verifiable answer. Avoid any ambiguity or room for interpretation.

    Relevance: The question MUST be directly applicable to assessing knowledge of the subject matter. The question should cover topics that can be objectively tested.

    Difficulty: The question should NOT be trivially easy. Assume the test-taker has basic knowledge of the subject matter. The ideal question assesses a slightly more nuanced understanding.

    No Subjectivity: The question MUST NOT rely on personal opinions, beliefs, or values. Avoid questions that involve "best practices" where multiple valid answers exist. Avoid hypothetical scenarios that require judgment calls.

    Clear and Concise Language: Use precise and unambiguous language. The question should be easy to understand and free from jargon or technical terms that are not essential.

    Format: Your output MUST be in the following format:

    """
