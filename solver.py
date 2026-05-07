"""
Agentic Project Euler solver — multi-provider LLM benchmark.

Uses native tool calling (function calling) for each provider.
The agent writes Python code, executes it locally, reads output,
debugs errors, and iterates until it submits an answer.

Usage:
    python solver.py --model gemini-2.5-flash --problem 1
    python solver.py --model gpt-5.4-mini --limit 50
    python solver.py                          # interactive
"""

import argparse
import csv
import json
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
    "gemini-3.1-flash-lite":  ("gemini",    "gemini-3.1-flash-lite-preview"),
    "gemini-3.1-pro":         ("gemini",    "gemini-3.1-pro-preview"),
    "gemini-2.5-flash":       ("gemini",    "gemini-2.5-flash"),
    "gemini-2.5-pro":         ("gemini",    "gemini-2.5-pro-preview-05-06"),
    "gpt-5.4-mini":           ("openai",    "gpt-5.4-mini"),
    "gpt-5.4-nano":           ("openai",    "gpt-5.4-nano"),
    "gpt-4o-mini":            ("openai",    "gpt-4o-mini"),
    "o3-mini":                ("openai",    "o3-mini"),
    "claude-sonnet-4-6":      ("anthropic", "claude-sonnet-4-6"),
    "claude-haiku-4.5":       ("anthropic", "claude-haiku-4-5-20251001"),
    "grok-3-mini":            ("xai",       "grok-3-mini"),
}

COST_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite-preview": (0.10, 0.40),
    "gemini-3.1-pro-preview":  (3.50, 10.50),
    "gemini-2.5-flash":        (0.15, 0.60),
    "gemini-2.5-pro-preview-05-06": (1.25, 10.00),
    "gpt-5.4-mini":    (0.40, 1.60),
    "gpt-5.4-nano":    (0.10, 0.40),
    "gpt-4o-mini":     (0.15, 0.60),
    "o3-mini":         (1.10, 4.40),
    "claude-sonnet-4-6":       (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "grok-3-mini":     (0.30, 0.50),
}

def _get_key(provider: str) -> str:
    key_map = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
               "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}
    val = os.getenv(key_map[provider])
    if not val:
        print(f"ERROR: {key_map[provider]} not set")
        sys.exit(1)
    return val

# ── ANSI ────────────────────────────────────────────────────────────────────

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[96m"; YELLOW = "\033[93m"; GREEN = "\033[92m"; RED = "\033[91m"

# ── Workspace & tool implementations ────────────────────────────────────────

WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_workspace")
os.makedirs(WORKSPACE, exist_ok=True)


def _exec_tool(name: str, args: dict) -> str:
    if name == "write_file":
        path = os.path.join(WORKSPACE, args.get("filename", "solve.py"))
        content = args.get("content", "")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {args.get('filename', 'solve.py')}"

    elif name == "run_python":
        fname = args.get("filename", "solve.py")
        path = os.path.join(WORKSPACE, fname)
        if not os.path.exists(path):
            return f"Error: {fname} does not exist"
        try:
            r = subprocess.run(
                [sys.executable, fname], capture_output=True, text=True,
                timeout=30, cwd=WORKSPACE, encoding="utf-8", errors="replace",
            )
            out = r.stdout
            if r.stderr:
                out += "\n[STDERR]\n" + r.stderr
            return out.strip() or "[No output]"
        except subprocess.TimeoutExpired:
            return "Error: script timed out (30s limit). Optimize your algorithm."
        except Exception as e:
            return f"Error: {e}"

    elif name == "submit_answer":
        return args.get("answer", "")

    return f"Unknown tool: {name}"


# ── Tool schema (provider-agnostic) ─────────────────────────────────────────

TOOL_DEFS = [
    {
        "name": "write_file",
        "description": "Write a Python script to the workspace. Always use this before run_python.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "File name, e.g. solve.py"},
                "content": {"type": "string", "description": "The Python source code"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "run_python",
        "description": "Execute a Python script and return stdout/stderr. Times out after 30s.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Script to run, e.g. solve.py"},
            },
            "required": ["filename"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit your final answer. You MUST call write_file and run_python first. "
            "The answer must be the exact output of your code. Do not guess."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The final numerical answer"},
            },
            "required": ["answer"],
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert mathematician and programmer solving Project Euler problems.
You have tools to write Python scripts, execute them, and submit answers.

RULES:
- You MUST write code and run it before submitting. Do NOT guess from memory.
- Scripts time out after 30 seconds — use efficient algorithms.
- The answer is always a single integer or number.
- All Python stdlib is available (math, itertools, functools, collections, etc.).
- Think about the math first, then write efficient code.
"""

MAX_STEPS = 15

# ── Provider-specific solve loops ───────────────────────────────────────────


def _retry(fn, *a, **kw):
    for attempt in range(3):
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e).lower()
            if attempt == 2 or not any(w in msg for w in (
                "429", "rate limit", "overloaded", "unavailable",
                "resource_exhausted", "500", "503")):
                raise
            time.sleep(2 ** (attempt + 1))


def solve(problem_text: str, provider: str, model: str, api_key: str):
    """
    Solve a problem using native tool calling.
    Returns (answer, steps, log, tokens_in, tokens_out).
    """
    if provider == "openai":
        return _solve_openai(problem_text, model, api_key)
    elif provider == "xai":
        return _solve_openai(problem_text, model, api_key, base_url="https://api.x.ai/v1")
    elif provider == "anthropic":
        return _solve_anthropic(problem_text, model, api_key)
    elif provider == "gemini":
        return _solve_gemini(problem_text, model, api_key)
    raise ValueError(f"Unknown provider: {provider}")


# ── OpenAI / xAI ────────────────────────────────────────────────────────────

def _solve_openai(problem_text, model, api_key, base_url=None):
    import openai
    kw = {"api_key": api_key}
    if base_url:
        kw["base_url"] = base_url
    client = openai.OpenAI(**kw)

    tools = [{"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["parameters"],
    }} for t in TOOL_DEFS]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this Project Euler problem:\n\n{problem_text}"},
    ]
    log = []
    tok_in = tok_out = 0

    for step in range(1, MAX_STEPS + 1):
        r = _retry(client.chat.completions.create,
                    model=model, messages=messages, tools=tools, temperature=0)
        if r.usage:
            tok_in += r.usage.prompt_tokens or 0
            tok_out += r.usage.completion_tokens or 0
        msg = r.choices[0].message

        if not msg.tool_calls:
            # No tool call — treat text as thinking, ask to use tools
            log.append({"step": step, "action": "text", "content": (msg.content or "")[:300]})
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": "Use the tools to write code and solve the problem. Call submit_answer when done."})
            print(f"  {DIM}Step {step}: text (no tool call){RESET}")
            continue

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result = _exec_tool(name, args)

            if name == "submit_answer":
                ans = args.get("answer", "").strip()
                wrote = any(e.get("action") == "write_file" for e in log)
                ran = any(e.get("action") == "run_python" for e in log)
                method = "code" if (wrote and ran) else "direct"
                log.append({"step": step, "action": "submit_answer", "answer": ans, "method": method})
                print(f"  {GREEN}Step {step}: submit_answer = {ans}  [{method}]{RESET}")
                return ans, step, log, tok_in, tok_out

            _log_tool(step, name, args, result, log)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "FAILED", MAX_STEPS, log, tok_in, tok_out


# ── Anthropic ────────────────────────────────────────────────────────────────

def _solve_anthropic(problem_text, model, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    tools = [{"name": t["name"], "description": t["description"],
              "input_schema": t["parameters"]} for t in TOOL_DEFS]

    messages = [{"role": "user", "content": f"Solve this Project Euler problem:\n\n{problem_text}"}]
    log = []
    tok_in = tok_out = 0

    for step in range(1, MAX_STEPS + 1):
        r = _retry(client.messages.create,
                    model=model, max_tokens=8192, temperature=0,
                    system=SYSTEM_PROMPT, messages=messages, tools=tools)
        if r.usage:
            tok_in += r.usage.input_tokens or 0
            tok_out += r.usage.output_tokens or 0

        if r.stop_reason != "tool_use":
            text = next((b.text for b in r.content if b.type == "text"), "")
            log.append({"step": step, "action": "text", "content": text[:300]})
            messages.append({"role": "assistant", "content": r.content})
            messages.append({"role": "user", "content": "Use the tools to write code and solve the problem. Call submit_answer when done."})
            print(f"  {DIM}Step {step}: text (no tool call){RESET}")
            continue

        messages.append({"role": "assistant", "content": r.content})
        tool_results = []
        for block in r.content:
            if block.type != "tool_use":
                continue
            name = block.name
            args = dict(block.input)
            result = _exec_tool(name, args)

            if name == "submit_answer":
                ans = args.get("answer", "").strip()
                wrote = any(e.get("action") == "write_file" for e in log)
                ran = any(e.get("action") == "run_python" for e in log)
                method = "code" if (wrote and ran) else "direct"
                log.append({"step": step, "action": "submit_answer", "answer": ans, "method": method})
                print(f"  {GREEN}Step {step}: submit_answer = {ans}  [{method}]{RESET}")
                return ans, step, log, tok_in, tok_out

            _log_tool(step, name, args, result, log)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    return "FAILED", MAX_STEPS, log, tok_in, tok_out


# ── Gemini ───────────────────────────────────────────────────────────────────

def _solve_gemini(problem_text, model, api_key):
    from google import genai
    from google.genai import types as gt

    client = genai.Client(api_key=api_key)

    _TYPE_MAP = {"string": gt.Type.STRING, "integer": gt.Type.INTEGER}
    decls = []
    for t in TOOL_DEFS:
        props = {}
        for pname, pinfo in t["parameters"]["properties"].items():
            props[pname] = gt.Schema(type=_TYPE_MAP.get(pinfo["type"], gt.Type.STRING),
                                     description=pinfo.get("description", ""))
        decls.append(gt.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters=gt.Schema(type=gt.Type.OBJECT, properties=props,
                                 required=t["parameters"].get("required", [])),
        ))

    config = gt.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[gt.Tool(function_declarations=decls)],
        temperature=0,
    )
    chat = client.chats.create(model=model, config=config)
    log = []
    tok_in = tok_out = 0

    response = _retry(chat.send_message, f"Solve this Project Euler problem:\n\n{problem_text}")

    for step in range(1, MAX_STEPS + 1):
        try:
            um = response.usage_metadata
            if um:
                tok_in += um.prompt_token_count or 0
                tok_out += um.candidates_token_count or 0
        except Exception:
            pass

        parts = response.candidates[0].content.parts or []
        fn_calls = [p.function_call for p in parts if p.function_call and p.function_call.name]

        if not fn_calls:
            text = response.text or ""
            log.append({"step": step, "action": "text", "content": text[:300]})
            response = _retry(chat.send_message,
                              "Use the tools to write code and solve the problem. Call submit_answer when done.")
            print(f"  {DIM}Step {step}: text (no tool call){RESET}")
            continue

        response_parts = []
        for fc in fn_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            result = _exec_tool(name, args)

            if name == "submit_answer":
                ans = args.get("answer", "").strip()
                wrote = any(e.get("action") == "write_file" for e in log)
                ran = any(e.get("action") == "run_python" for e in log)
                method = "code" if (wrote and ran) else "direct"
                log.append({"step": step, "action": "submit_answer", "answer": ans, "method": method})
                print(f"  {GREEN}Step {step}: submit_answer = {ans}  [{method}]{RESET}")
                return ans, step, log, tok_in, tok_out

            _log_tool(step, name, args, result, log)
            response_parts.append(gt.Part.from_function_response(name=name, response={"result": result}))

        response = _retry(chat.send_message, response_parts)

    return "FAILED", MAX_STEPS, log, tok_in, tok_out


# ── Shared logging ──────────────────────────────────────────────────────────

def _log_tool(step, name, args, result, log):
    if name == "write_file":
        log.append({"step": step, "action": "write_file",
                     "code": args.get("content", ""), "result": result})
    elif name == "run_python":
        log.append({"step": step, "action": "run_python", "output": result[:500]})
    elif name == "read_file":
        log.append({"step": step, "action": "read_file", "result": result[:500]})
    preview = result[:120] + ("..." if len(result) > 120 else "")
    print(f"  {YELLOW}Step {step}: {name}{RESET}  {DIM}{preview}{RESET}")


# ── Problem fetching ────────────────────────────────────────────────────────

def fetch_problem(n: int) -> str | None:
    import requests
    try:
        r = requests.get(f"https://projecteuler.net/minimal={n}", timeout=10)
        return r.text.strip() if r.status_code == 200 else None
    except Exception:
        return None


def load_solutions(path: str = None) -> dict[int, str]:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Solutions.md")
    sols = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"(\d+)\.\s+(\S+)", line.strip())
                if m:
                    sols[int(m.group(1))] = m.group(2)
    except FileNotFoundError:
        pass
    return sols


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic Project Euler Solver")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--problem", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

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

    print(f"\n{CYAN}{BOLD}Euler Solver  ({model_key} via {provider}){RESET}")
    print(f"{'─' * 50}")

    solutions = load_solutions()

    if args.problem:
        problem_nums = [args.problem]
    else:
        problem_nums = sorted(solutions.keys())
        if args.limit:
            problem_nums = problem_nums[:args.limit]

    results = []
    total_tok_in = total_tok_out = 0

    for pn in problem_nums:
        text = fetch_problem(pn)
        if not text:
            print(f"\n{RED}Problem {pn}: could not fetch{RESET}")
            continue

        expected = solutions.get(pn)
        print(f"\n{BOLD}Problem {pn}{RESET}" +
              (f"  {DIM}(expected: {expected}){RESET}" if expected else ""))

        t0 = time.time()
        answer, steps, log, tok_in, tok_out = solve(text, provider, api_model, api_key)
        elapsed = time.time() - t0
        total_tok_in += tok_in
        total_tok_out += tok_out

        correct = (answer == expected) if expected else None
        method = log[-1].get("method", "?") if log else "?"
        mark = f"{GREEN}PASS{RESET}" if correct else (
               f"{RED}FAIL{RESET}" if correct is False else f"{DIM}?{RESET}")

        print(f"  {mark}  answer={answer}  steps={steps}  method={method}  time={elapsed:.1f}s")
        results.append({
            "problem": pn, "answer": answer, "expected": expected or "",
            "correct": correct, "steps": steps, "time_s": round(elapsed, 1),
            "method": method,
        })

    if results:
        tested = [r for r in results if r["correct"] is not None]
        passed = sum(1 for r in tested if r["correct"])
        total_time = sum(r["time_s"] for r in results)
        rates = COST_PER_1M.get(api_model, (0, 0))
        cost = (total_tok_in * rates[0] + total_tok_out * rates[1]) / 1_000_000

        print(f"\n{'=' * 50}")
        print(f"  {BOLD}{model_key}{RESET}: {passed}/{len(tested)} correct  "
              f"({total_time:.1f}s total)")
        print(f"  Tokens: {total_tok_in:,} in / {total_tok_out:,} out  (~${cost:.4f})")
        print(f"{'=' * 50}\n")

        csv_path = f"results_{model_key.replace('.', '_')}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"  Saved to {csv_path}")


if __name__ == "__main__":
    main()
