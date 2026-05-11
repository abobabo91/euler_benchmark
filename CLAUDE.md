# euler_benchmark

Agentic LLM benchmark on Project Euler problems. Each model gets a sandboxed workspace with three tools (`write_file`, `run_python`, `submit_answer`) and up to 15 iterations to solve a problem. Multi-provider (Gemini, OpenAI, Anthropic, xAI). Flask web UI + CLI.

## Critical rules

- The model **NEVER sees the expected answer** during a run. Comparison is post-hoc only. Don't surface ground-truth into agent context — that would invalidate every result.
- Answers must come from **code execution**, not direct guessing. The `submit_answer` tool is one-shot, no take-backs. Direct guesses (no code run) are flagged. Don't relax this.
- NEVER commit any API key. Users enter keys in the web UI; nothing is persisted to the repo. Local `.env` is gitignored.
- Result JSONs (`results_<model>.json`) are part of the public leaderboard — only commit when they reflect honest runs.

## Stack & run

- Python + Flask web app + multi-provider SDKs
- Web: `pip install -r requirements.txt && python app.py` → http://localhost:5050
- CLI: `python solver.py --model <model> --problem <N>` (or `--limit`, `--problems 1,2,3`)
- Docker: `Dockerfile` present for containerized runs
- Step budget: 15 iterations per problem (with warnings before the cap)
- Per-script timeout: 30 seconds on `run_python`

## Where things live

- Web app entry: `app.py` (Flask, port 5050)
- CLI entry: `solver.py`
- Templates: `templates/` (dark-themed leaderboard)
- Problem data + expected answers: `data/`
- Solutions notes / reference: `Solutions.md`
- Per-model results: `results_<model>.json` at repo root

## Common workflows

**To benchmark a new model:**
1. Add the model name + provider routing in `solver.py` (or wherever the model dispatch lives)
2. Add the API key field in the Flask UI form
3. Run a small slice first: `python solver.py --model <new> --limit 5` to verify auth + tool-use works
4. Then run a larger range; results stream into `results_<new>.json`
5. Commit only `results_*.json` for completed honest runs (per Critical rule)

**To inspect why a model failed a problem:**
1. Find the per-problem log in the web UI (click the result row) — shows every step: code written, execution output, debugging
2. The full log lives in the run artifact saved per problem

## Gotchas

- **Cost varies wildly by model.** As of May 2026 README leaderboard: claude-opus-4-7 ran 2 problems for $9.96 ($5/problem); gemini-3.1-flash-lite ran 118 for $0.42. Cost-aware sampling matters. For cross-project model selection: `tools/knowledge base/llms 2026.md`.
- Some problems require >30s of compute. Models that don't optimize their code time out and have to iterate. This is by design — efficient algorithms are part of the score.
- Tool-use API shape differs per provider: Gemini native function calling, OpenAI tool use, Anthropic tool use, xAI OpenAI-compatible. The dispatch logic in `solver.py` handles this — don't bypass it when adding a new provider.
- 15-iteration step limit is intentional. Without it, models loop on verification forever. Don't raise without a strong reason.
