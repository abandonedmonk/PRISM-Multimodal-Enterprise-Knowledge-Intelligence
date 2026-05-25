from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

try:
	import config
except ImportError:
	config = None

from langgraph.graph import StateGraph, END

from agents.state import CorpMindState, AgentMessage
from agents.tools.retrieval import retrieve_chunks, retrieve_combined
from agents.tools.graph_traversal import graph_local_context, graph_global_context
from agents.tools.web_search import search_web
from agents.tools.vision import analyze_vision

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

DEFAULT_FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/gpt-oss-120b"


def _get_client() -> OpenAI:
	api_key = (
		(config.FIREWORKS_API_KEY if config else "")
		or os.getenv("FIREWORKS_API_KEY", "")
		or (config.LLM_API_KEY if config else "")
		or os.getenv("LLM_API_KEY", "")
		or "not-needed"
	)
	base_url = (
		(config.FIREWORKS_BASE_URL if config else "")
		or os.getenv("FIREWORKS_BASE_URL", DEFAULT_FIREWORKS_BASE)
	)
	if not base_url or base_url == DEFAULT_BASE_URL:
		base_url = DEFAULT_FIREWORKS_BASE
	return OpenAI(api_key=api_key, base_url=base_url)


def _get_model() -> str:
	return (
		(config.FIREWORKS_MODEL if config else "")
		or os.getenv("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL)
	)


def _route_query(state: CorpMindState) -> dict:
	explicit_mode = state.get("retrieval_mode")
	if explicit_mode and explicit_mode != "auto":
		return {"route": explicit_mode}

	query = (state.get("query") or "").lower()

	if state.get("use_vision") or any(k in query for k in ("chart", "table", "image", "figure")):
		route = "vision"
	elif state.get("use_web") or any(k in query for k in ("latest", "news", "web", "online")):
		route = "web"
	elif any(k in query for k in ("trend", "theme", "overall", "global")):
		route = "graph_global"
	elif any(k in query for k in ("relationship", "related", "connected", "impact")):
		route = "graph_local"
	else:
		route = "combined"

	return {"route": route}


def _retrieve_context(state: CorpMindState) -> dict:
	route = state.get("route", "combined")
	top_k = int(state.get("top_k") or 5)
	trace = list(state.get("tool_trace", []))
	metadata_filter = state.get("metadata_filter")

	if route == "combined":
		trace.append("combined")
		data = retrieve_combined(query=state.get("query", ""), top_k=top_k, metadata_filter=metadata_filter)
	elif route == "graph_global":
		trace.append("graph_global")
		data = graph_global_context(state.get("query", ""), top_k=top_k)
	elif route == "graph_local":
		trace.append("graph_local")
		data = graph_local_context(state.get("query", ""), top_k=top_k, hop=2)
	elif route == "vector":
		trace.append("vector")
		data = retrieve_chunks(state.get("query", ""), top_k=top_k, hybrid=False, metadata_filter=metadata_filter)
	elif route == "hybrid":
		trace.append("hybrid")
		data = retrieve_chunks(state.get("query", ""), top_k=top_k, hybrid=True, metadata_filter=metadata_filter)
	else:
		trace.append("combined")
		data = retrieve_combined(query=state.get("query", ""), top_k=top_k, metadata_filter=metadata_filter)

	stats = data.get("stats", {})
	return {
		"context_text": data.get("context_text", ""),
		"sources": data.get("sources", []),
		"tool_trace": trace,
		"context_stats": stats,
	}


def _web_context(state: CorpMindState) -> dict:
	trace = list(state.get("tool_trace", []))
	trace.append("web_search")
	data = search_web(state.get("query", ""), top_k=int(state.get("top_k") or 5))
	note = data.get("note")
	context_text = data.get("context_text", "")
	if note and not context_text:
		context_text = note
	return {
		"context_text": context_text,
		"sources": data.get("sources", []),
		"tool_trace": trace,
		"context_stats": {"web_results": 1},
	}


def _vision_context(state: CorpMindState) -> dict:
	trace = list(state.get("tool_trace", []))
	trace.append("vision")
	data = analyze_vision(state.get("query", ""))
	note = data.get("note")
	context_text = data.get("context_text", "")
	if note and not context_text:
		context_text = note
	return {
		"context_text": context_text,
		"sources": data.get("sources", []),
		"tool_trace": trace,
		"context_stats": {"vision_results": 1},
	}


def _answer(state: CorpMindState) -> dict:
	client = _get_client()
	model = _get_model()
	temperature = float(state.get("temperature") or 0.2)

	system_prompt = (
		"You are a helpful financial analyst assistant. Use only the provided context to answer the question. "
		"If the context is empty or insufficient, say you do not have enough information. "
		"Cite specific details from the context when relevant."
	)

	context_text = (state.get("context_text") or "").strip()
	query = state.get("query") or ""

	user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}"

	messages: List[dict] = [{"role": "system", "content": system_prompt}]
	for msg in state.get("chat_history", []):
		if msg.get("role") and msg.get("content"):
			messages.append({"role": msg["role"], "content": msg["content"]})
	messages.append({"role": "user", "content": user_prompt})

	resp = client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=temperature,
		max_tokens=1024,
	)
	if not resp.choices:
		return {"answer": ""}

	msg = resp.choices[0].message
	content = msg.content

	if content is None and hasattr(msg, 'model_extra'):
		content = msg.model_extra.get("reasoning_content", "") or msg.model_extra.get("thought", "")

	return {"answer": (content or "").strip()}


def _route_to_node(state: CorpMindState) -> str:
	route = state.get("route", "combined")
	if route == "web":
		return "web_search"
	if route == "vision":
		return "vision"
	return "retrieve"


def build_agent():
	graph = StateGraph(CorpMindState)

	graph.add_node("route", _route_query)
	graph.add_node("retrieve", _retrieve_context)
	graph.add_node("web_search", _web_context)
	graph.add_node("vision", _vision_context)
	graph.add_node("answer", _answer)

	graph.set_entry_point("route")
	graph.add_conditional_edges("route", _route_to_node)

	graph.add_edge("retrieve", "answer")
	graph.add_edge("web_search", "answer")
	graph.add_edge("vision", "answer")
	graph.add_edge("answer", END)

	return graph.compile()


def run_agent(
	query: str,
	chat_history: List[AgentMessage] | None = None,
	top_k: int = 5,
	use_web: bool = False,
	use_vision: bool = False,
	retrieval_mode: str = "auto",
	metadata_filter=None,
	temperature: float = 0.2,
) -> CorpMindState:
	app = build_agent()
	state: CorpMindState = {
		"query": query,
		"chat_history": chat_history or [],
		"top_k": top_k,
		"use_web": use_web,
		"use_vision": use_vision,
		"retrieval_mode": retrieval_mode,
		"metadata_filter": metadata_filter,
		"temperature": temperature,
	}
	return app.invoke(state)
