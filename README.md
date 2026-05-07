# Euler Solver

Agentic LLM benchmark on Project Euler problems. Each model gets a workspace where it can write Python code, execute it, debug errors, and iterate until it finds the answer. No internet access — pure computation and algorithmic reasoning.

## How it works

The agent receives a Project Euler problem and has three tools:
- **write_file** — write a Python script to the workspace
- **run_python** — execute it (30s timeout)
- **read_file** — read a file back

The LLM must reason about the math, write efficient code, run it, and return the answer. Scripts that time out force the model to think about algorithmic complexity.

## Web UI

A Flask web app with a dark-themed UI for running benchmarks interactively:

- Enter API keys for any provider (Gemini, OpenAI, Anthropic, xAI)
- Pick a problem number and select models
- Watch results come in live with pass/fail, steps, time, and cost
- Leaderboard tracks accuracy across all runs

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5050
```

## CLI

```bash
# Solve a single problem
python solver.py --model gemini-2.5-flash --problem 1

# Solve first 10 problems
python solver.py --model gpt-4o-mini --limit 10

# Interactive model picker
python solver.py
```

## Supported models

| Provider | Models |
|----------|--------|
| Gemini | gemini-2.5-flash, gemini-2.5-pro, gemini-2.5-flash-lite |
| OpenAI | gpt-4o-mini, gpt-4o, o3-mini |
| Anthropic | claude-sonnet-4-6, claude-haiku-3.5 |
| xAI | grok-3-mini |

## Results (Problem 1)

| Model | Steps | Time | Cost | Result |
|-------|-------|------|------|--------|
| claude-haiku-3.5 | 1 | 5.5s | $0.0025 | PASS |
| gemini-2.5-flash | 2 | 19.8s | $0.0004 | PASS |
| gpt-4o-mini | 3 | 9.9s | $0.0002 | PASS |
| grok-3-mini | 3 | 27.2s | $0.0005 | PASS |
