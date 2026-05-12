# Euler Benchmark

**Live:** https://euler-benchmark-673626542594.europe-west1.run.app (Cloud Run, project `zeta-matrix-483109-u9`, region `europe-west1`, service `euler-benchmark`)

An agentic LLM benchmark on Project Euler problems. Each model gets a sandboxed workspace where it can write Python code, execute it, debug errors, and iterate until it submits an answer. The model **never sees the correct answer** -- comparison happens post-hoc.

## How it works

```
Problem text --> LLM Agent --> write_file() --> run_python() --> debug --> submit_answer()
                    ^                                |
                    |________________________________|
                          (up to 15 iterations)
```

The agent has three tools:
- **write_file** -- write a Python script to the workspace
- **run_python** -- execute it (30s timeout, stdout/stderr returned)
- **submit_answer** -- commit a final answer (one shot, no take-backs)

The LLM must reason about the math, write efficient algorithms, run them, and submit. Scripts that time out force the model to optimize. All steps, code, and outputs are logged for transparency.

**Anti-cheating measures:**
- The model never sees the expected answer at any point
- All answers must come from code execution (direct guesses are flagged)
- Full agent logs are saved and inspectable per problem
- Step limits with warnings force commitment rather than infinite verification loops

## Web UI

A Flask app with dark-themed leaderboard for running benchmarks interactively:

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5050
```

- Enter API keys for any provider (Gemini, OpenAI, Anthropic, xAI)
- Select models and problem ranges
- Watch results come in with pass/fail, steps, time, and cost
- Leaderboard tracks accuracy across all runs
- Click any result to inspect full agent logs (code written, execution output, debugging steps)

## CLI

```bash
# Solve a single problem
python solver.py --model gemini-3.1-flash-lite --problem 1

# Solve problems 1-50
python solver.py --model gpt-5.4-mini --limit 50

# Retry specific failures
python solver.py --model gemini-3.1-flash-lite --problems 51,55,60

# Interactive model picker
python solver.py
```

## Supported models

| Provider | Models | Notes |
|----------|--------|-------|
| Google | gemini-3.1-flash-lite, gemini-3.1-pro, gemini-2.5-flash, gemini-2.5-pro | Native function calling |
| OpenAI | gpt-5.4-mini, gpt-5.4-nano, gpt-4o-mini, o3-mini | Tool use API |
| Anthropic | claude-sonnet-4-6, claude-haiku-4.5 | Tool use API |
| xAI | grok-3-mini | OpenAI-compatible |

## Results

> I'm too broke to run all models on all 100 problems, but the partial results are already interesting.

### Leaderboard (as of May 2026)

| Model | Problems Tested | Correct | Accuracy | Cost | Avg Time/Problem |
|-------|:-:|:-:|:-:|:-:|:-:|
| **claude-opus-4-7** | 2 | 2 | **100.0%** | $9.96 | 4.7min |
| **claude-haiku-4.5** | 5 | 5 | **100.0%** | $0.15 | 39.7s |
| **gemini-3.1-flash-lite** | 118 | 111 | **94.1%** | $0.42 | ~10s |
| **gemini-3.1-pro** | 2 | 1 | **50.0%** | $1.32 | 8.4min |
| **gpt-5.4-mini** | 5 | 4 | **80.0%** | $0.01 | 8.9s |
| **gpt-5.4** | 2 | 0 | **0.0%** | $0.19 | 2.2min |
| **grok-3-mini** | 5 | 3 | **60.0%** | $0.01 | 118.1s |

### Flagship showdown (P685 + P562)

Two hard problems (difficulty 500+) tested across flagship models:

| Model | P685 (digit sums) | P562 (geometry) | Cost |
|-------|:--:|:--:|:--:|
| **claude-opus-4-7** | PASS | PASS | $9.96 |
| gemini-3.1-pro | PASS | FAIL | $1.32 |
| gpt-5.4 | FAIL | CRASH | $0.19 |
| gemini-3.1-flash-lite | FAIL | FAIL | $0.04 |

Opus is the only model that solved both. GPT-5.4 crashed on P562 due to OpenAI's content policy flagging a geometry problem mid-conversation.

### Key findings

**gemini-3.1-flash-lite** was tested on P1-141 (116 problems). At $0.38 total, it's extremely cost-effective. Accuracy degrades with difficulty: 100% on P1-50, 96% on P51-100, 76% on P101-141. Its main weakness is over-verification: it often found the right answer early but kept rewriting code to "make sure," burning through its 15-step limit without submitting. Adding a step counter to every tool response (`[Step 5/15]`) significantly reduced this behavior.

**claude-haiku-4.5** solved all 5 hard problems perfectly, including P84 (Monopoly odds via Markov chains) which stumped both GPT and Grok. It writes careful, well-structured code and verifies with multiple approaches. Most expensive per problem but most reliable.

**gpt-5.4-mini** is the fastest -- averaging 3 steps per problem. It solved P74 and P90 that Gemini couldn't, but missed P84. Very cost-effective at $0.013 for 5 problems.

**grok-3-mini** was the slowest (heavy "thinking" overhead) and guessed directly on P84 and P95 without running code, getting both wrong. Its reasoning model architecture adds latency without improving accuracy on these computational problems.

### Failure analysis

Problems that tripped models up:

| Problem | Difficulty | Flash-Lite | GPT-5.4-mini | Haiku-4.5 | Grok-3-mini |
|---------|-----------|:---:|:---:|:---:|:---:|
| P74 (digit factorial chains) | Medium | FAIL | PASS | PASS | PASS |
| P84 (Monopoly Markov chain) | Hard | PASS | FAIL | PASS | FAIL |
| P90 (cube digit pairs) | Hard | FAIL | PASS | PASS | PASS |
| P93 (arithmetic expressions) | Hard | PASS | PASS | PASS | PASS |
| P95 (amicable chains) | Hard | PASS | PASS | PASS | FAIL |

P84 is the hardest -- it requires implementing a full Monopoly simulation with Community Chest, Chance cards, and jail mechanics, then computing a Markov chain steady-state distribution. Only Claude got it right.

### Cheating verification

Looking at the agent logs confirms genuine problem-solving:
- Every correct answer came from code that was written, executed, and produced the output
- Models frequently debug errors (off-by-one, timeouts, wrong algorithms) and iterate
- The write-run-debug loop is visible in the logs -- this isn't memorization
- Problems requiring external data files (P54, P67, P81-83, P89) show the model downloading and parsing data

## Project structure

```
euler_benchmark/
  solver.py          # CLI solver with native tool calling per provider
  app.py             # Flask web UI
  Solutions.md       # Known answers (never shown to the model)
  requirements.txt   # Python dependencies
  data/
    results.json     # All benchmark results with full agent logs
  templates/
    index.html       # Web UI template
```

## Running your own benchmarks

1. Clone this repo
2. Set API keys: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`
3. `pip install -r requirements.txt`
4. `python solver.py --model <model> --limit 50`

Results are saved incrementally -- if a run crashes, you keep everything up to that point.

## License

MIT
