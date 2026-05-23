""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Entity Extraction (Placeholder)                                           ║
    ║  Reserved for future named entity recognition pipelines.                   ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Placeholder module for future NER (Named Entity Recognition) integration.
    Will extract company names, dates, financial metrics, etc. from filings.

Future Integration:
    - spaCy with financial domain models
    - FinBERT for financial entity extraction
    - Custom patterns for ticker symbols, CIK numbers, amounts

Status:
    NOT YET IMPLEMENTED - awaiting domain-specific NER model selection
"""


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text.
    
    Args:
        text: Input text from filing chunk
        
    Returns:
        List of dicts with entity type, value, position, confidence
        
    Raises:
        NotImplementedError: Awaiting NER model integration
        
    Future Example:
        entities = extract_entities("Apple Inc. reported Q4 revenue of $119.6B")
        # Returns: [
        #   {"type": "ORG", "value": "Apple Inc.", "start": 0, "end": 10, "score": 0.99},
        #   {"type": "AMOUNT", "value": "$119.6B", "start": ..., ...}
        # ]
    """
	
raise NotImplementedError("Entity extraction pipeline not implemented yet")
