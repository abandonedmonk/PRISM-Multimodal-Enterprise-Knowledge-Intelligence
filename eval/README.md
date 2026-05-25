# RAGAS Evaluation

This folder contains a local-first eval runner for PRISM question records.

## Input format

Use JSON or JSONL with records shaped like:

```json
{
  "ticker": "AAPL",
  "domain": "Revenue",
  "question_type": "Direct",
  "filing": "2024 10-K",
  "question": "What was Apple's total net sales for fiscal year 2024 ...?",
  "answer": "Apple's total net sales for FY2024 were $391.035 billion ..."
}
```

Required fields:
- `ticker`
- `domain`
- `question_type`
- `filing`
- `question`
- `answer`

## Run

Minimal run (uses defaults):

```bash
python eval/evaluate.py --input eval/test_set.json
```

Same run with explicit options (equivalent to defaults):

```bash
python eval/evaluate.py \
  --input eval/test_set.json \
  --output eval/results.json \
  --phase all \
  --workers 1 \
  --retrieval-mode combined \
  --top-k 5 \
  --temperature 0.0 \
  --max-tokens 512
```

Default values:
- `--output`: `eval/results.json`
- `--phase`: `all`
- `--workers`: `1`
- `--retrieval-mode`: `combined`
- `--top-k`: `5`
- `--use-domain-as-section`: `false` (off unless flag is passed)
- `--limit`: `0` (no limit)
- `--temperature`: `0.0`
- `--max-tokens`: `512`
- `--generator-base-url`: `PRISM_EVAL_GENERATOR_BASE_URL` env var or built-in default
- `--generator-model`: `PRISM_EVAL_GENERATOR_MODEL` env var or built-in default
- `--generator-api-key`: `PRISM_EVAL_GENERATOR_API_KEY` or `LLM_API_KEY` or `"not-needed"`
- `--judge-base-url`: `PRISM_EVAL_JUDGE_BASE_URL` env var or built-in default
- `--judge-model`: `PRISM_EVAL_JUDGE_MODEL` env var or built-in default
- `--judge-api-key`: `PRISM_EVAL_JUDGE_API_KEY` or `FIREWORKS_API_KEY` or empty string
- `--embedding-model`: `PRISM_EVAL_EMBEDDING_MODEL` env var or `BAAI/bge-small-en-v1.5`

Run only retrieval+generation (skip RAGAS):

```bash
python eval/evaluate.py \
  --input eval/test_set.json \
  --output eval/results/generated_only.json \
  --phase generate \
  --workers 6
```

Run RAGAS later on an existing generated report:

```bash
python eval/evaluate.py \
  --phase ragas \
  --results-input eval/results/generated_only.json \
  --output eval/results/generated_plus_ragas.json
```

## Output

The report includes:
- row-level generated answers
- per-row RAGAS metrics (`faithfulness`, `answer_relevancy`, `answer_correctness`) when available
- retrieval, generation, and total latency
- aggregate RAGAS summary and per-metric coverage counts
- grouped summaries by ticker, domain, question type, and filing
- a final top-level `summary` block for quick reporting
- supports phased execution (`all`, `generate`, `ragas`) and parallel generation workers

## Notes

- The judge defaults to Fireworks GPT OSS (`accounts/fireworks/models/gpt-oss-120b`).
- Use `--judge-provider minimax` to switch to Minimax (`abab6.5s-chat`) and set `MINIMAX_API_KEY` in your environment, or `--judge-provider custom` with explicit `--judge-base-url` and `--judge-model`.
- If you want to use Modal or another provider later, override `--generator-base-url` and `--judge-base-url`.
