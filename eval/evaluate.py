#!/usr/bin/env python3
"""Local-first RAGAS evaluation harness for PRISM question records.

Expected input format per record:
{
  "ticker": "AAPL",
  "domain": "Revenue",
  "question_type": "Direct",
  "filing": "2024 10-K",
  "question": "...",
  "answer": "..."
}

Supports JSON arrays and JSONL files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import statistics
import time
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

try:
	import config
except ImportError:
	config = None


SYSTEM_PROMPT = (
	"You are a helpful financial analyst assistant. Use only the provided context to answer the question. "
	"If the context is empty or insufficient, say you do not have enough information. "
	"Cite specific details from the context when relevant."
)

DEFAULT_GENERATOR_BASE_URL = "https://anshujena007--prism-llm-serve-serve.modal.run/v1"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_JUDGE_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _load_json_records(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		raise FileNotFoundError(f"Input file not found: {path}")

	raw = path.read_text(encoding="utf-8").strip()
	if not raw:
		return []

	if path.suffix.lower() in {".jsonl", ".ndjson"}:
		records = []
		for line_no, line in enumerate(raw.splitlines(), start=1):
			line = line.strip()
			if not line:
				continue
			try:
				records.append(json.loads(line))
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
		return records

	payload = json.loads(raw)
	if isinstance(payload, list):
		return payload
	if isinstance(payload, dict):
		for key in ("records", "questions", "items", "data", "test_set"):
			value = payload.get(key)
			if isinstance(value, list):
				return value
		return [payload]

	raise ValueError(f"Unsupported JSON payload in {path}")


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
	ticker = str(raw.get("ticker", "")).strip().upper()
	domain = str(raw.get("domain", "")).strip()
	question_type = str(raw.get("question_type", "")).strip()
	filing = str(raw.get("filing", "")).strip()
	question = str(raw.get("question", "")).strip()
	answer = str(raw.get("answer", "")).strip()

	if not question:
		raise ValueError("Each record must include a non-empty 'question' field.")
	if not answer:
		raise ValueError("Each record must include a non-empty 'answer' field.")

	return {
		"ticker": ticker,
		"domain": domain,
		"question_type": question_type,
		"filing": filing,
		"question": question,
		"answer": answer,
	}


def _parse_filing_label(filing: str) -> dict[str, Any]:
	filing = filing.strip()
	year_match = re.search(r"(19|20)\d{2}", filing)
	form_match = re.search(r"10-[KQ]", filing, flags=re.IGNORECASE)
	quarter_match = re.search(r"Q[1-4]", filing, flags=re.IGNORECASE)

	parsed: dict[str, Any] = {}
	if year_match:
		parsed["year"] = int(year_match.group(0))
	if form_match:
		parsed["filing_type"] = form_match.group(0).upper()
	if quarter_match:
		parsed["quarter"] = quarter_match.group(0).upper()
	return parsed


def _build_metadata_filter(record: dict[str, Any], use_domain_as_section: bool) -> dict[str, Any]:
	metadata: dict[str, Any] = {"ticker": record["ticker"]}
	metadata.update(_parse_filing_label(record["filing"]))
	if use_domain_as_section and record.get("domain"):
		metadata["section"] = record["domain"]
	return {k: v for k, v in metadata.items() if v is not None and v != ""}


def _split_context(context_text: str) -> list[str]:
	parts = [part.strip() for part in context_text.split("\n\n---\n\n")]
	return [part for part in parts if part]


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
	return OpenAI(api_key=api_key, base_url=base_url)


def _generate_answer(
	*,
	client: OpenAI,
	model: str,
	question: str,
	context_text: str,
	temperature: float,
	max_tokens: int,
) -> str:
	response = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{question}"},
		],
		temperature=temperature,
		max_tokens=max_tokens,
	)
	if not response.choices:
		return ""
	message = response.choices[0].message
	content = message.content or ""
	if not content and hasattr(message, "model_extra"):
		content = (
			message.model_extra.get("reasoning_content", "")
			or message.model_extra.get("thought", "")
			or ""
		)
	return content.strip()


def _retrieve_context(
	*,
	question: str,
	metadata_filter: dict[str, Any],
	retrieval_mode: str,
	top_k: int,
) -> dict[str, Any]:
	from agents.tools.graph_traversal import graph_global_context, graph_local_context
	from agents.tools.retrieval import retrieve_chunks, retrieve_combined

	if retrieval_mode == "combined":
		data = retrieve_combined(question, top_k=top_k, metadata_filter=metadata_filter)
		return {
			"context_text": data.get("context_text", ""),
			"sources": data.get("sources", []),
			"stats": data.get("stats", {}),
		}

	if retrieval_mode == "hybrid":
		data = retrieve_chunks(question, top_k=top_k, hybrid=True, metadata_filter=metadata_filter)
		return {
			"context_text": data.get("context_text", ""),
			"sources": data.get("sources", []),
			"stats": {"chunks": len(data.get("chunks", []))},
		}

	if retrieval_mode == "vector":
		data = retrieve_chunks(question, top_k=top_k, hybrid=False, metadata_filter=metadata_filter)
		return {
			"context_text": data.get("context_text", ""),
			"sources": data.get("sources", []),
			"stats": {"chunks": len(data.get("chunks", []))},
		}

	if retrieval_mode == "graph_local":
		data = graph_local_context(question, top_k=top_k, hop=2)
		return {
			"context_text": data.get("context_text", ""),
			"sources": data.get("sources", []),
			"stats": {"chunk_ids": len(data.get("chunk_ids", []))},
		}

	if retrieval_mode == "graph_global":
		data = graph_global_context(question, top_k=top_k)
		return {
			"context_text": data.get("context_text", ""),
			"sources": data.get("sources", []),
			"stats": {"chunk_ids": len(data.get("chunk_ids", []))},
		}

	raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


class FastEmbedEmbeddings:
	def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
		from fastembed import TextEmbedding

		self._model = TextEmbedding(model_name)

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		embeddings = []
		for item in self._model.embed(texts):
			embeddings.append(item.tolist())
		return embeddings

	def embed_query(self, text: str) -> list[float]:
		for item in self._model.embed([text]):
			return item.tolist()
		return []



def _build_ragas_wrappers(
	*,
	judge_base_url: str,
	judge_model: str,
	judge_api_key: str,
	embedding_model: str,
):
	judge_llm = None
	judge_embeddings = None

	try:
		from langchain_openai import ChatOpenAI

		judge_llm = ChatOpenAI(
			base_url=judge_base_url,
			api_key=judge_api_key,
			model=judge_model,
			temperature=0,
		)
	except Exception as exc:
		raise RuntimeError(f"Could not build judge LLM client: {exc}") from exc

	try:
		judge_embeddings = FastEmbedEmbeddings(model_name=embedding_model)
	except Exception:
		judge_embeddings = None

	try:
		from ragas.llms import LangchainLLMWrapper

		judge_llm = LangchainLLMWrapper(judge_llm)
	except Exception:
		pass

	try:
		from ragas.embeddings import LangchainEmbeddingsWrapper

		if judge_embeddings is not None:
			judge_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)
	except Exception:
		pass

	return judge_llm, judge_embeddings


def _compute_ragas_scores(rows: list[dict[str, Any]], args) -> dict[str, Any]:
	try:
		from datasets import Dataset
		from ragas import evaluate
		from ragas.metrics import answer_correctness, answer_relevancy, faithfulness
	except Exception as exc:
		return {"available": False, "reason": f"RAGAS import failed: {exc}"}

	judge_llm, judge_embeddings = _build_ragas_wrappers(
		judge_base_url=args.judge_base_url,
		judge_model=args.judge_model,
		judge_api_key=args.judge_api_key,
		embedding_model=args.embedding_model,
	)

	metric_list = [faithfulness, answer_relevancy, answer_correctness]
	metric_names = [metric.name for metric in metric_list]

	ragas_rows = []
	for row in rows:
		ragas_rows.append(
			{
				"question": row["question"],
				"answer": row["generated_answer"],
				"contexts": row["contexts"],
				"ground_truth": row["reference_answer"],
			}
		)

	dataset = Dataset.from_list(ragas_rows)
	try:
		result = evaluate(
			dataset=dataset,
			metrics=metric_list,
			llm=judge_llm,
			embeddings=judge_embeddings,
		)
	except Exception as exc:
		return {"available": False, "reason": f"RAGAS evaluation failed: {exc}"}

	if hasattr(result, "to_pandas"):
		df = result.to_pandas()
	elif isinstance(result, pd.DataFrame):
		df = result
	else:
		df = pd.DataFrame(result)

	metric_columns = [col for col in metric_names if col in df.columns]
	per_row_metrics: list[dict[str, Any]] = []
	if metric_columns:
		for _, score_row in df[metric_columns].iterrows():
			item: dict[str, Any] = {}
			for metric in metric_columns:
				value = score_row.get(metric)
				if pd.notna(value):
					item[metric] = float(value)
				else:
					item[metric] = None
			per_row_metrics.append(item)

	if len(per_row_metrics) < len(rows):
		per_row_metrics.extend({metric: None for metric in metric_columns} for _ in range(len(rows) - len(per_row_metrics)))
	elif len(per_row_metrics) > len(rows):
		per_row_metrics = per_row_metrics[: len(rows)]

	summary: dict[str, float] = {}
	for metric in metric_columns:
		series = pd.to_numeric(df[metric], errors="coerce").dropna()
		if not series.empty:
			summary[metric] = float(series.mean())

	coverage = {
		metric: int(sum(1 for row_metric in per_row_metrics if row_metric.get(metric) is not None))
		for metric in metric_columns
	}

	return {
		"available": True,
		"summary": summary,
		"per_row_metrics": per_row_metrics,
		"metric_columns": metric_columns,
		"coverage": coverage,
	}


def _percentile(values: list[float], percentile: float) -> float:
	if not values:
		return 0.0
	ordered = sorted(values)
	if len(ordered) == 1:
		return float(ordered[0])
	rank = (len(ordered) - 1) * (percentile / 100.0)
	low = int(rank)
	high = min(low + 1, len(ordered) - 1)
	if low == high:
		return float(ordered[low])
	fraction = rank - low
	return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def _summarize_latencies(rows: list[dict[str, Any]]) -> dict[str, float]:
	retrieval = [float(row["retrieval_ms"]) for row in rows]
	generation = [float(row["generation_ms"]) for row in rows]
	total = [float(row["latency_ms"]) for row in rows]
	return {
		"retrieval_p50_ms": _percentile(retrieval, 50),
		"retrieval_p95_ms": _percentile(retrieval, 95),
		"generation_p50_ms": _percentile(generation, 50),
		"generation_p95_ms": _percentile(generation, 95),
		"latency_p50_ms": _percentile(total, 50),
		"latency_p95_ms": _percentile(total, 95),
		"retrieval_avg_ms": float(statistics.mean(retrieval)) if retrieval else 0.0,
		"generation_avg_ms": float(statistics.mean(generation)) if generation else 0.0,
		"latency_avg_ms": float(statistics.mean(total)) if total else 0.0,
	}


def _group_summary(rows: list[dict[str, Any]], metric_columns: list[str]) -> list[dict[str, Any]]:
	if not rows:
		return []

	frame = pd.DataFrame(rows)
	group_columns = [col for col in ("ticker", "domain", "question_type", "filing") if col in frame.columns]
	if not group_columns:
		return []

	aggregated = []
	for keys, subset in frame.groupby(group_columns, dropna=False):
		if not isinstance(keys, tuple):
			keys = (keys,)
		item = {column: value for column, value in zip(group_columns, keys)}
		for metric in metric_columns:
			metric_values = [value for value in subset["metrics"].apply(lambda x: (x or {}).get(metric)).tolist() if value is not None]
			if metric_values:
				item[metric] = float(statistics.mean(metric_values))
			else:
				item[metric] = None
		item["count"] = int(len(subset))
		item["latency_ms_p50"] = _percentile(subset["latency_ms"].astype(float).tolist(), 50)
		item["latency_ms_p95"] = _percentile(subset["latency_ms"].astype(float).tolist(), 95)
		aggregated.append(item)

	return sorted(aggregated, key=lambda row: (-row["count"], row.get("ticker", ""), row.get("filing", "")))


import math


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Infinity with None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def _save_results_json(report: dict[str, Any], args) -> dict[str, Any]:
	"""Save evaluation results to the requested output path."""
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	sanitized = _sanitize_for_json(report)
	output_path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False), encoding="utf-8")

	return {"saved": True, "path": str(output_path.resolve())}


def _run_generation_row(record: dict[str, Any], index: int, args) -> dict[str, Any]:
	metadata_filter = _build_metadata_filter(record, args.use_domain_as_section)
	generator_client = _get_openai_client(args.generator_base_url, args.generator_api_key)

	retrieval_started = time.perf_counter()
	retrieval = _retrieve_context(
		question=record["question"],
		metadata_filter=metadata_filter,
		retrieval_mode=args.retrieval_mode,
		top_k=args.top_k,
	)
	retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

	generation_started = time.perf_counter()
	generated_answer = _generate_answer(
		client=generator_client,
		model=args.generator_model,
		question=record["question"],
		context_text=retrieval["context_text"],
		temperature=args.temperature,
		max_tokens=args.max_tokens,
	)
	generation_ms = (time.perf_counter() - generation_started) * 1000.0

	contexts = _split_context(retrieval["context_text"])
	if not contexts and retrieval["context_text"]:
		contexts = [retrieval["context_text"]]

	row = {
		**record,
		"row_index": index,
		"generated_answer": generated_answer,
		"reference_answer": record["answer"],
		"contexts": contexts,
		"context_count": len(contexts),
		"source_count": len(retrieval.get("sources", [])),
		"retrieval_mode": args.retrieval_mode,
		"retrieval_ms": retrieval_ms,
		"generation_ms": generation_ms,
		"latency_ms": retrieval_ms + generation_ms,
		"context_text": retrieval["context_text"],
		"sources": retrieval.get("sources", []),
		"stats": retrieval.get("stats", {}),
		"metrics": {},
	}
	return row


def _run_generation_phase(records: list[dict[str, Any]], args) -> list[dict[str, Any]]:
	if args.workers <= 1:
		results = []
		for index, record in enumerate(records, start=1):
			row = _run_generation_row(record, index, args)
			results.append(row)
			print(
				f"[gen {index}/{len(records)}] {record['ticker']} {record['filing']} "
				f"{record['question_type']} | {row['latency_ms']:.1f} ms"
			)
		return results

	results_map: dict[int, dict[str, Any]] = {}
	with ThreadPoolExecutor(max_workers=args.workers) as executor:
		futures = {
			executor.submit(_run_generation_row, record, index, args): (index, record)
			for index, record in enumerate(records, start=1)
		}
		for future in as_completed(futures):
			index, record = futures[future]
			row = future.result()
			results_map[index] = row
			print(
				f"[gen {index}/{len(records)}] {record['ticker']} {record['filing']} "
				f"{record['question_type']} | {row['latency_ms']:.1f} ms"
			)
	return [results_map[i] for i in sorted(results_map.keys())]


def _build_report_from_results(results: list[dict[str, Any]], args, input_path: str) -> dict[str, Any]:
	ragas_result: dict[str, Any]
	if args.phase in {"all", "ragas"}:
		ragas_result = _compute_ragas_scores(results, args)
		if ragas_result.get("available") and ragas_result.get("per_row_metrics"):
			for row, metrics in zip(results, ragas_result["per_row_metrics"]):
				row["metrics"] = metrics
	else:
		ragas_result = {
			"available": False,
			"summary": {},
			"metric_columns": [],
			"coverage": {},
			"reason": "RAGAS not run for phase=generate.",
		}

	metric_columns = ragas_result.get("metric_columns", [])
	summary = {
		"records": len(results),
		"phase": args.phase,
		"ragas_available": ragas_result.get("available", False),
		"ragas_metrics": ragas_result.get("summary", {}),
		"ragas_coverage": ragas_result.get("coverage", {}),
		"latency": _summarize_latencies(results),
	}

	return {
		"config": {
			"input": input_path,
			"phase": args.phase,
			"workers": args.workers,
			"retrieval_mode": args.retrieval_mode,
			"top_k": args.top_k,
			"generator_base_url": args.generator_base_url,
			"generator_model": args.generator_model,
			"judge_base_url": args.judge_base_url,
			"judge_model": args.judge_model,
			"embedding_model": args.embedding_model,
			"use_domain_as_section": args.use_domain_as_section,
		},
		"count": len(results),
		"latency": summary["latency"],
		"ragas": {
			"available": ragas_result.get("available", False),
			"summary": ragas_result.get("summary", {}),
			"metric_columns": metric_columns,
			"coverage": ragas_result.get("coverage", {}),
			"reason": ragas_result.get("reason"),
		},
		"results": results,
		"by_group": _group_summary(results, metric_columns),
		"summary": summary,
	}


def run_eval(args) -> dict[str, Any]:
	if args.phase == "ragas":
		if not args.results_input:
			raise ValueError("For --phase ragas, provide --results-input with a generated eval report JSON.")
		report_input = Path(args.results_input)
		data = json.loads(report_input.read_text(encoding="utf-8"))
		results = data.get("results")
		if not isinstance(results, list) or not results:
			raise ValueError(f"No results rows found in {report_input}")
		return _build_report_from_results(results, args, str(report_input.resolve()))

	if not args.input:
		raise ValueError("For --phase all or --phase generate, provide --input.")

	raw_records = _load_json_records(Path(args.input))
	if args.limit:
		raw_records = raw_records[: args.limit]

	records = [_normalize_record(record) for record in raw_records]
	if not records:
		raise ValueError("No evaluation records found.")
	results = _run_generation_phase(records, args)
	return _build_report_from_results(results, args, str(Path(args.input).resolve()))


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run a local RAGAS evaluation over PRISM question records.")
	parser.add_argument("--input", default="", help="Path to a JSON or JSONL file with evaluation records.")
	parser.add_argument(
		"--results-input",
		default="",
		help="Existing eval report JSON with a `results` array. Required when --phase ragas.",
	)
	parser.add_argument("--output", default="eval/results.json", help="Where to write the full JSON report.")
	parser.add_argument(
		"--phase",
		default="all",
		choices=["all", "generate", "ragas"],
		help="all=generate+ragas, generate=only retrieval+generation, ragas=score an existing results report.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=1,
		help="Parallel workers for retrieval+generation phase.",
	)
	parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per question.")
	parser.add_argument(
		"--retrieval-mode",
		default="combined",
		choices=["combined", "hybrid", "vector", "graph_local", "graph_global"],
		help="Retrieval path used to build context.",
	)
	parser.add_argument(
		"--use-domain-as-section",
		action="store_true",
		help="Map the domain field onto the section metadata filter.",
	)
	parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of rows to evaluate.")
	parser.add_argument(
		"--generator-base-url",
		default=os.getenv("PRISM_EVAL_GENERATOR_BASE_URL", DEFAULT_GENERATOR_BASE_URL),
		help="OpenAI-compatible base URL for answer generation.",
	)
	parser.add_argument(
		"--generator-model",
		default=os.getenv("PRISM_EVAL_GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL),
		help="Model name for answer generation.",
	)
	parser.add_argument(
		"--generator-api-key",
		default=os.getenv("PRISM_EVAL_GENERATOR_API_KEY", os.getenv("LLM_API_KEY", "not-needed")),
		help="API key for the generator endpoint.",
	)
	parser.add_argument(
		"--judge-base-url",
		default=os.getenv("PRISM_EVAL_JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
		help="Base URL for the RAGAS judge model.",
	)
	parser.add_argument(
		"--judge-model",
		default=os.getenv("PRISM_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
		help="Model name for the RAGAS judge.",
	)
	parser.add_argument(
		"--judge-api-key",
		default=os.getenv("PRISM_EVAL_JUDGE_API_KEY", os.getenv("FIREWORKS_API_KEY", "")),
		help="API key for the judge endpoint.",
	)
	parser.add_argument(
		"--embedding-model",
		default=os.getenv("PRISM_EVAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
		help="FastEmbed model used for RAGAS embedding metrics.",
	)
	parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
	parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens for generation.")
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	report = run_eval(args)
	saved = _save_results_json(report, args)

	print("\nSummary")
	print(f"  Records: {report['count']}")
	print(f"  Latency p50: {report['latency']['latency_p50_ms']:.1f} ms")
	print(f"  Latency p95: {report['latency']['latency_p95_ms']:.1f} ms")

	if report["ragas"]["available"]:
		print("  RAGAS metrics:")
		for key, value in report["ragas"]["summary"].items():
			print(f"    {key}: {value:.4f}")
	else:
		print(f"  RAGAS unavailable: {report['ragas'].get('reason', 'unknown reason')}")

	print(f"\nWrote report to {saved['path']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
