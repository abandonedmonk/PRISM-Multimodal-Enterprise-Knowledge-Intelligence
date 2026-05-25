from typing import TypedDict, List, Dict, Optional, Any


class AgentMessage(TypedDict):
	role: str
	content: str


class CorpMindState(TypedDict, total=False):
	query: str
	route: str
	chat_history: List[AgentMessage]
	context_text: str
	sources: List[Dict[str, Any]]
	answer: str
	tool_trace: List[str]
	top_k: int
	use_web: bool
	use_vision: bool
	retrieval_mode: str
	metadata_filter: Optional[Dict[str, Any]]
	temperature: float
	context_stats: Optional[Dict[str, int]]
