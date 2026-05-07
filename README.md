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
python solver.py --model gemini-3.1-flash-lite --problem 1

# Solve first 10 problems
python solver.py --model gpt-5.4-mini --limit 10

# Interactive model picker
python solver.py
```

## Supported models

| Provider | Models |
|----------|--------|
| Gemini | gemini-3.1-flash-lite, gemini-3.1-pro, gemini-2.5-flash, gemini-2.5-pro |
| OpenAI | gpt-5.4-mini, gpt-5.4-nano, gpt-4o-mini, o3-mini |
| Anthropic | claude-sonnet-4-6, claude-haiku-4.5 |
| xAI | grok-3-mini |

## Results

Tested on 3 difficulty tiers across 4 newest fast models. 12/12 pass.

### Problem 1 (easy) — sum of multiples of 3 or 5 below 1000

| Model | Steps | Time | Cost |
|-------|-------|------|------|
| gpt-5.4-mini | 1 | 4.8s | $0.0002 |
| gemini-3.1-flash-lite | 3 | 7.8s | $0.0002 |
| claude-sonnet-4-6 | 3 | 7.3s | $0.0061 |
| grok-3-mini | 3 | 27.6s | $0.0005 |

### Problem 14 (medium) — longest Collatz sequence under 1M

| Model | Steps | Time | Cost |
|-------|-------|------|------|
| gemini-3.1-flash-lite | 3 | 9.5s | $0.0004 |
| claude-sonnet-4-6 | 3 | 10.2s | $0.0119 |
| gpt-5.4-mini | 3 | 10.4s | $0.0011 |
| grok-3-mini | 3 | 69.8s | $0.0007 |

### Problem 12 (hard) — first triangle number with 500+ divisors

| Model | Steps | Time | Cost |
|-------|-------|------|------|
| gpt-5.4-mini | 3 | 9.6s | $0.0016 |
| gemini-3.1-flash-lite | 3 | 10.9s | $0.0004 |
| claude-sonnet-4-6 | 3 | 12.9s | $0.0115 |
| grok-3-mini | 3 | 45.1s | $0.0008 |
