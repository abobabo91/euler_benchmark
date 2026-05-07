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

Tested across 5 difficulty tiers. Models must write and execute code — direct guessing is flagged.

### Summary

| Model | P1 | P12 | P14 | P100 | P300 | Score |
|-------|:--:|:---:|:---:|:----:|:----:|:-----:|
| gemini-3.1-flash-lite | PASS | PASS | PASS | PASS | FAIL | 4/5 |
| claude-sonnet-4-6 | PASS | PASS | PASS | PASS | FAIL | 4/5 |
| gpt-5.4-mini | PASS | PASS | PASS | FAIL | FAIL | 3/5 |
| grok-3-mini | PASS | PASS | PASS | FAIL | FAIL | 3/5 |

### P1 (easy) — multiples of 3/5

All 4 pass. gpt-5.4-mini answered in 1 step without code (memorized).

### P14 (medium) — Collatz sequences

All 4 pass. All wrote code. Grok slowest at 69.8s.

### P12 (hard) — triangle number with 500+ divisors

All 4 pass. All wrote efficient divisor-counting algorithms.

### P100 (very hard) — Pell equation / Diophantine

| Model | Result | Method | Time | Cost |
|-------|--------|--------|------|------|
| gemini-3.1-flash-lite | PASS | code | 10.5s | $0.0006 |
| claude-sonnet-4-6 | PASS | code (with verification) | 34.2s | $0.0621 |
| gpt-5.4-mini | FAIL | code (wrong algorithm) | 10.8s | $0.0014 |
| grok-3-mini | FAIL | direct guess (wrong) | 83.5s | $0.0001 |

### P300 (extreme) — 2D protein folding

All 4 fail. Claude tried hardest (15 steps, 7 rewrites, $0.62 spent). Gemini hit timeouts and iterated. GPT and Grok gave up without writing code.
