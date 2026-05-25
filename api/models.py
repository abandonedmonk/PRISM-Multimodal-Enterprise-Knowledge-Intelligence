from typing import Optional
from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    ticker: Optional[str] = Field(None, description="Filter by ticker symbol (e.g., AAPL, NVDA)")
    filing_type: Optional[str] = Field(None, description="Filter by filing type (e.g., 10-K, 10-Q)")
    year: Optional[int] = Field(None, description="Filter by year (e.g., 2024)")
    quarter: Optional[str | int] = Field(None, description="Filter by quarter (e.g., Q3, Q4)")
    section: Optional[str] = Field(None, description="Filter by section (e.g., item1, item7)")
    company_name: Optional[str] = Field(None, description="Filter by company name (partial match)")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    retrieval_mode: str = Field(
        default="combined",
        description="Retrieval mode: combined, hybrid, graph_local, graph_global, vector"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results per retrieval path")
    chat_history: list[dict] = Field(default_factory=list, description="Chat history for context")
    metadata_filter: Optional[MetadataFilter] = Field(None, description="Filter results by metadata")
    use_web: bool = Field(default=False, description="Enable web search")
    use_vision: bool = Field(default=False, description="Enable vision analysis")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM temperature")
    stream: bool = Field(default=False, description="Stream response (not implemented yet)")


class SourceItem(BaseModel):
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str | int] = None
    chunk_id: Optional[str] = None
    section: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None


class ContextUsed(BaseModel):
    qdrant_chunks: int = 0
    graph_entities: int = 0
    communities: int = 0
    total_sources: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    tool_trace: list[str]
    context_used: ContextUsed
    retrieval_mode: str
    latency_ms: float


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field(default="combined", description="combined, hybrid, vector, graph_local, graph_global")
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filter: Optional[MetadataFilter] = None


class RetrievalResponse(BaseModel):
    chunks: list[dict]
    graph_context: Optional[dict] = None
    sources: list[SourceItem]
    retrieval_mode: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    neo4j: str
    fireworks: str
    embedding: str