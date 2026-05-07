"""
Agentic Project Euler solver — multi-provider LLM benchmark.

The agent writes Python code, executes it locally, reads output,
debugs errors, and iterates until it finds the answer. No internet
access — pure computation and algorithmic reasoning.

Usage:
    python solver.py --model gemini-2.5-flash
    python solver.py --model gpt-4o-mini --limit 10
    python solver.py --model claude-sonnet-4-6 --problem 1
    python solver.py                          # interactive model picker
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

from dotenv import load_dotenv

# Unicode safety for Windows
for _s in ("stdout", "stderr"):
    _stream = getattr(sys, _s, None)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

load_dotenv()

# ── Model registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # Gemini
    "gemini-3.1-flash-lite":         ("gemini",    "gemini-3.1-flash-lite-preview"),
    "gemini-3.1-pro":                ("gemini",    "gemini-3.1-pro-preview"),
    "gemini-2.5-flash":              ("gemini",    "gemini-2.5-flash"),
    "gemini-2.5-pro":                ("gemini",    "gemini-2.5-pro-preview-05-06"),
    # OpenAI
    "gpt-5.4-mini":                  ("openai",    "gpt-5.4-mini"),
    "gpt-5.4-nano":                  ("openai",    "gpt-5.4-nano"),
    "gpt-4o-mini":                   ("openai",    "gpt-4o-mini"),
    "o3-mini":                       ("openai",    "o3-mini"),
    # Anthropic
    "claude-sonnet-4-6":             ("anthropic", "claude-sonnet-4-6"),
    "claude-haiku-4.5":              ("anthropic", "claude-haiku-4-5-20251001"),
    # xAI
    "grok-3-mini":                   ("xai",       "grok-3-mini"),
}

COST_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite-preview": (0.10,  0.40),
    "gemini-3.1-pro-preview":        (3.50, 10.50),
    "gemini-2.5-flash":              (0.15,  0.60),
    "gemini-2.5-pro-preview-05-06":  (1.25, 10.00),
    "gpt-5.4-mini":          (0.40,   1.60),
    "gpt-5.4-nano":          (0.10,   0.40),
    "gpt-4o-mini":           (0.15,   0.60),
    "o3-mini":               (1.10,   4.40),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001":  (0.80,  4.00),
    "grok-3-mini":           (0.30,   0.50),
}

# ── API key resolution ──────────────────────────────────────────────────────

def _get_key(provider: str) -> str:
    key_map = {
        "gemini":    "GEMINI_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai":       "XAI_API_KEY",
    }
    val = os.getenv(key_map[provider])
    if not val:
        print(f"ERROR: {key_map[provider]} not set in .env")
        sys.exit(1)
    return val


# ── ANSI colours ────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
MAGENTA = "\033[95m"

# ── Workspace ───────────────────────────────────────────────────────────────

WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_workspace")
os.makedirs(WORKSPACE, exist_ok=True)


def write_file(filename: str, content: str) -> str:
    path = os.path.join(WORKSPACE, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {filename}"


def read_file(filename: str) -> str:
    path = os.path.join(WORKSPACE, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: {filename} not found"


def run_python(filename: str) -> str:
    """Execute a Python script with a 30s timeout. Returns stdout+stderr."""
    path = os.path.join(WORKSPACE, filename)
    if not os.path.exists(path):
        return f"Error: {filename} does not exist"
    try:
        r = subprocess.run(
            [sys.executable, filename],
            capture_output=True, text=True, timeout=30,
            cwd=WORKSPACE, encoding="utf-8", errors="replace",
        )
        out = r.stdout
        if r.stderr:
            out += "\n[STDERR]\n" + r.stderr
        return out.strip() or "[No output]"
    except subprocess.TimeoutExpired:
        return "Error: script timed out (30s limit). Optimize your algorithm."
    except Exception as e:
        return f"Error: {e}"


TOOLS = {"write_file": write_file, "read_file": read_file, "run_python": run_python}

# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert mathematician and programmer solving Project Euler problems.
You have a local Python workspace with three tools:

1. write_file(filename, content) — write a Python script
2. run_python(filename) — execute it and see stdout/stderr
3. read_file(filename) — read a file back

WORKFLOW:
- Think about the math first. Identify the pattern or formula.
- Write efficient Python code (avoid brute force when a formula exists).
- Run it. If it errors or times out, read the error, fix, and retry.
- ONLY submit final_answer AFTER you have written code, run it, and confirmed the output.

RULES:
- You MUST write and execute code to compute the answer. Do NOT guess or recall answers from memory.
- You have NO internet access. Pure computation only.
- Scripts time out after 30 seconds — use efficient algorithms.
- The answer is always a single integer or number.
- Use math, itertools, functools, collections — all stdlib is available.
- For large problems, think about complexity before coding.
- Your final_answer must match the exact output of your code. No guessing.

TOOL FORMAT — respond with exactly one JSON block per message:

To write a file:
```json
{"tool": "write_file", "filename": "solve.py", "content": "print(sum(range(10)))"}
```

To run a file:
```json
{"tool": "run_python", "filename": "solve.py"}
```

To submit the final answer:
```json
{"final_answer": "12345"}
```

Output ONLY the JSON block. No extra text outside it.
"""

# ── LLM chat backends ──────────────────────────────────────────────────────

class LLMChat:
    """Uniform chat interface across providers."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.tokens_in = 0
        self.tokens_out = 0

    def send(self, messages: list[dict]) -> str:
        for attempt in range(3):
            try:
                return self._send(messages)
            except Exception as e:
                msg = str(e).lower()
                retryable = any(w in msg for w in (
                    "429", "rate limit", "overloaded", "unavailable", "timeout",
                    "resource_exhausted", "server", "500", "503",
                ))
                if not retryable or attempt == 2:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  {DIM}Retry {attempt+1}/3 in {wait}s ({e}){RESET}")
                time.sleep(wait)

    def _send(self, messages: list[dict]) -> str:
        if self.provider == "gemini":
            return self._gemini(messages)
        elif self.provider == "openai":
            return self._openai(messages)
        elif self.provider == "anthropic":
            return self._anthropic(messages)
        elif self.provider == "xai":
            return self._openai(messages, base_url="https://api.x.ai/v1")
        raise ValueError(f"Unknown provider: {self.provider}")

    def _openai(self, messages, base_url=None):
        import openai
        kw = {"api_key": self.api_key}
        if base_url:
            kw["base_url"] = base_url
        client = openai.OpenAI(**kw)
        r = client.chat.completions.create(
            model=self.model, messages=messages, temperature=0,
        )
        if r.usage:
            self.tokens_in += r.usage.prompt_tokens or 0
            self.tokens_out += r.usage.completion_tokens or 0
        return r.choices[0].message.content

    def _anthropic(self, messages):
        import anthropic
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        msgs = [m for m in messages if m["role"] != "system"]
        client = anthropic.Anthropic(api_key=self.api_key)
        r = client.messages.create(
            model=self.model, max_tokens=4096, temperature=0,
            system=system, messages=msgs,
        )
        if r.usage:
            self.tokens_in += r.usage.input_tokens or 0
            self.tokens_out += r.usage.output_tokens or 0
        return r.content[0].text

    def _gemini(self, messages):
        from google import genai
        from google.genai import types as gt
        client = genai.Client(api_key=self.api_key)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(gt.Content(role=role, parts=[gt.Part(text=m["content"])]))
        config = gt.GenerateContentConfig(
            system_instruction=system, temperature=0,
        )
        r = client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        try:
            um = r.usage_metadata
            if um:
                self.tokens_in += um.prompt_token_count or 0
                self.tokens_out += um.candidates_token_count or 0
        except Exception:
            pass
        return r.text

    def cost(self) -> float:
        rates = COST_PER_1M.get(self.model, (0, 0))
        return (self.tokens_in * rates[0] + self.tokens_out * rates[1]) / 1_000_000


# ── JSON parsing ────────────────────────────────────────────────────────────

def parse_json(text: str | None) -> dict | None:
    """Extract a JSON object from LLM output (handles code fences)."""
    if not text:
        return None
    # Try code-fenced JSON first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    m = re.search(r"\{[^{}]*\"(?:tool|final_answer)\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Try the largest {...} block
    for m in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and ("tool" in d or "final_answer" in d):
                return d
        except json.JSONDecodeError:
            continue
    return None


# ── Agent loop ──────────────────────────────────────────────────────────────

MAX_STEPS = 15


def solve(problem_text: str, chat: LLMChat) -> tuple[str, int, list[dict]]:
    """
    Run the agent loop. Returns (answer, steps_used, log).
    Log contains each step: tool calls, code written, outputs.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this Project Euler problem:\n\n{problem_text}"},
    ]
    log: list[dict] = []

    for step in range(1, MAX_STEPS + 1):
        response = chat.send(messages) or ""
        messages.append({"role": "assistant", "content": response})

        data = parse_json(response)
        if data is None:
            log.append({"step": step, "action": "parse_error", "raw": response[:500]})
            messages.append({"role": "user", "content":
                "No valid JSON found. Respond with exactly one JSON block: "
                "either a tool call or {\"final_answer\": \"NUMBER\"}."})
            continue

        # Final answer
        if "final_answer" in data:
            ans = str(data["final_answer"]).strip()
            # Check if they wrote code or just guessed
            wrote_code = any(e.get("action") == "write_file" for e in log)
            ran_code = any(e.get("action") == "run_python" for e in log)
            method = "code" if (wrote_code and ran_code) else "direct"
            log.append({"step": step, "action": "final_answer", "answer": ans, "method": method})
            print(f"  {GREEN}Step {step}: final_answer = {ans}  [{method}]{RESET}")
            return ans, step, log

        # Tool call
        tool_name = data.get("tool")
        if tool_name not in TOOLS:
            log.append({"step": step, "action": "unknown_tool", "tool": tool_name})
            messages.append({"role": "user", "content":
                f"Unknown tool '{tool_name}'. Available: write_file, run_python, read_file."})
            continue

        # Execute
        if tool_name == "write_file":
            code = data.get("content", "")
            result = write_file(data.get("filename", "solve.py"), code)
            log.append({"step": step, "action": "write_file", "code": code, "result": result})
        elif tool_name == "run_python":
            result = run_python(data.get("filename", "solve.py"))
            log.append({"step": step, "action": "run_python", "output": result[:500]})
        elif tool_name == "read_file":
            result = read_file(data.get("filename", "solve.py"))
            log.append({"step": step, "action": "read_file", "result": result[:500]})

        # Log
        preview = result[:120] + ("..." if len(result) > 120 else "")
        print(f"  {YELLOW}Step {step}: {tool_name}{RESET}  {DIM}{preview}{RESET}")

        messages.append({"role": "user", "content": f"Tool output ({tool_name}):\n{result}"})

    return "FAILED", MAX_STEPS, log


# ── Problem fetching ────────────────────────────────────────────────────────

def fetch_problem(n: int) -> str | None:
    import requests
    try:
        r = requests.get(f"https://projecteuler.net/minimal={n}", timeout=10)
        return r.text.strip() if r.status_code == 200 else None
    except Exception:
        return None


def load_solutions(path: str = "Solutions.md") -> dict[int, str]:
    sols = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"(\d+)\.\s+(\S+)", line)
                if m:
                    sols[int(m.group(1))] = m.group(2)
    except FileNotFoundError:
        pass
    return sols


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic Project Euler Solver")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        help="LLM model to use")
    parser.add_argument("--problem", type=int, help="Solve a single problem number")
    parser.add_argument("--limit", type=int, help="Max problems to solve")
    args = parser.parse_args()

    # Model selection
    model_key = args.model
    if not model_key:
        keys = sorted(MODEL_REGISTRY.keys())
        print(f"\n{BOLD}Select model:{RESET}")
        for i, k in enumerate(keys, 1):
            prov, name = MODEL_REGISTRY[k]
            print(f"  {i}. {k}  ({prov}: {name})")
        try:
            idx = int(input(f"\nChoice (1-{len(keys)}): ")) - 1
            model_key = keys[idx]
        except (ValueError, IndexError):
            print("Invalid choice.")
            return

    provider, api_model = MODEL_REGISTRY[model_key]
    api_key = _get_key(provider)
    chat = LLMChat(provider, api_model, api_key)

    print(f"\n{CYAN}{BOLD}Euler Solver  ({model_key} via {provider}){RESET}")
    print(f"{'─' * 50}")

    solutions = load_solutions()

    # Determine which problems to solve
    if args.problem:
        problem_nums = [args.problem]
    else:
        problem_nums = sorted(solutions.keys())
        if args.limit:
            problem_nums = problem_nums[:args.limit]

    results = []
    for pn in problem_nums:
        text = fetch_problem(pn)
        if not text:
            print(f"\n{RED}Problem {pn}: could not fetch{RESET}")
            continue

        expected = solutions.get(pn)
        print(f"\n{BOLD}Problem {pn}{RESET}" +
              (f"  {DIM}(expected: {expected}){RESET}" if expected else ""))

        t0 = time.time()
        answer, steps, log = solve(text, chat)
        elapsed = time.time() - t0

        correct = (answer == expected) if expected else None
        method = log[-1].get("method", "?") if log else "?"
        mark = f"{GREEN}PASS{RESET}" if correct else (
               f"{RED}FAIL{RESET}" if correct is False else f"{DIM}?{RESET}")

        print(f"  {mark}  answer={answer}  steps={steps}  method={method}  time={elapsed:.1f}s")
        results.append({
            "problem": pn, "answer": answer, "expected": expected or "",
            "correct": correct, "steps": steps, "time_s": round(elapsed, 1),
        })

    # Summary
    if results:
        tested = [r for r in results if r["correct"] is not None]
        passed = sum(1 for r in tested if r["correct"])
        total_time = sum(r["time_s"] for r in results)
        cost = chat.cost()

        print(f"\n{'═' * 50}")
        print(f"  {BOLD}{model_key}{RESET}: {passed}/{len(tested)} correct  "
              f"({total_time:.1f}s total)")
        print(f"  Tokens: {chat.tokens_in:,} in / {chat.tokens_out:,} out"
              f"  (~${cost:.4f})")
        print(f"{'═' * 50}\n")

        # Save CSV
        import csv
        csv_path = f"results_{model_key.replace('.', '_')}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"  Saved to {csv_path}")


if __name__ == "__main__":
    main()
