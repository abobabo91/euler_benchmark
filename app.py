"""
Flask web app for the Euler Solver benchmark.
"""

import json
import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request

from solver import (
    MODEL_REGISTRY, COST_PER_1M, solve, fetch_problem, load_solutions,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
os.makedirs(DATA_DIR, exist_ok=True)

PROVIDER_KEY_MAP = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}


# ── Storage ──────────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_results(results: list[dict]):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def append_result(record: dict):
    results = load_results()
    results.append(record)
    save_results(results)


def load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


# ── Leaderboard ──────────────────────────────────────────────────────────────

def build_leaderboard() -> list[dict]:
    stats: dict[str, dict] = {}
    for r in load_results():
        mid = r.get("model_id", "")
        if mid not in stats:
            stats[mid] = {
                "model_id": mid,
                "runs": 0, "correct": 0,
                "total_steps": 0, "total_time": 0, "total_cost": 0,
            }
        s = stats[mid]
        s["runs"] += 1
        s["correct"] += 1 if r.get("correct") else 0
        s["total_steps"] += r.get("steps", 0)
        s["total_time"] += r.get("time_s", 0)
        s["total_cost"] += r.get("cost", 0)

    rows = []
    for s in stats.values():
        n = s["runs"] or 1
        rows.append({
            **s,
            "accuracy": round(s["correct"] / n * 100, 1),
            "avg_steps": round(s["total_steps"] / n, 1),
            "avg_time": round(s["total_time"] / n, 1),
            "avg_cost": round(s["total_cost"] / n, 5),
        })
    rows.sort(key=lambda r: (-r["accuracy"], r["avg_steps"], r["avg_time"]))
    return rows


# ── App ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)
runs: dict[str, dict] = {}
runs_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    config = load_config()
    models = []
    for key, (provider, api_model) in sorted(MODEL_REGISTRY.items()):
        env_key = PROVIDER_KEY_MAP[provider]
        has_key = bool(config.get(env_key, "").strip())
        models.append({
            "id": key,
            "provider": provider,
            "api_model": api_model,
            "available": has_key,
        })
    return jsonify(models)


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        config = load_config()
        config.update(request.json or {})
        save_config(config)
        return jsonify({"ok": True})
    config = load_config()
    return jsonify({
        "has_key": {k: bool(v) for k, v in config.items()},
        "masked": {
            k: (v[:6] + "..." + v[-4:] if len(v) > 10 else ("set" if v else ""))
            for k, v in config.items()
        },
    })


@app.route("/api/solutions")
def api_solutions():
    sols = load_solutions()
    return jsonify({str(k): v for k, v in sols.items()})


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.json or {}
    problem_num = payload.get("problem")
    model_ids = payload.get("models", [])

    if not problem_num or not model_ids:
        return jsonify({"error": "Missing problem or models"}), 400

    problem_num = int(problem_num)
    problem_text = fetch_problem(problem_num)
    if not problem_text:
        return jsonify({"error": f"Could not fetch problem {problem_num}"}), 400

    solutions = load_solutions()
    expected = solutions.get(problem_num)

    config = load_config()
    run_id = str(uuid.uuid4())[:8]

    model_results = {}
    for mid in model_ids:
        if mid not in MODEL_REGISTRY:
            continue
        provider, _ = MODEL_REGISTRY[mid]
        env_key = PROVIDER_KEY_MAP[provider]
        if not config.get(env_key):
            model_results[mid] = {"status": "error", "error": f"No API key for {provider}"}
            continue
        model_results[mid] = {
            "status": "pending",
            "steps": 0,
            "answer": "",
            "correct": None,
            "time_s": 0,
            "cost": 0,
            "log": [],
        }

    with runs_lock:
        runs[run_id] = {
            "problem": problem_num,
            "expected": expected,
            "model_results": model_results,
        }

    def worker(mid: str):
        provider, api_model = MODEL_REGISTRY[mid]
        env_key = PROVIDER_KEY_MAP[provider]
        api_key = config.get(env_key, "")

        with runs_lock:
            runs[run_id]["model_results"][mid]["status"] = "running"

        import time as _time
        t0 = _time.time()
        try:
            answer, steps, log, tok_in, tok_out = solve(problem_text, provider, api_model, api_key)
            elapsed = _time.time() - t0
            correct = (answer == expected) if expected else None
            rates = COST_PER_1M.get(api_model, (0, 0))
            cost = (tok_in * rates[0] + tok_out * rates[1]) / 1_000_000
            method = log[-1].get("method", "?") if log else "?"

            with runs_lock:
                runs[run_id]["model_results"][mid].update({
                    "status": "done",
                    "answer": answer,
                    "steps": steps,
                    "correct": correct,
                    "time_s": round(elapsed, 1),
                    "cost": round(cost, 5),
                    "tokens_in": tok_in,
                    "tokens_out": tok_out,
                    "method": method,
                    "log": log,
                })

            append_result({
                "model_id": mid,
                "problem": problem_num,
                "answer": answer,
                "expected": expected or "",
                "correct": correct,
                "steps": steps,
                "time_s": round(elapsed, 1),
                "cost": round(cost, 5),
                "method": method,
                "log": log,
            })

        except Exception as e:
            elapsed = _time.time() - t0
            with runs_lock:
                runs[run_id]["model_results"][mid].update({
                    "status": "error",
                    "error": str(e),
                    "time_s": round(elapsed, 1),
                })

    for mid in model_results:
        if model_results[mid].get("status") != "error":
            t = threading.Thread(target=worker, args=(mid,), daemon=True)
            t.start()

    return jsonify({"run_id": run_id})


@app.route("/api/status/<run_id>")
def api_status(run_id: str):
    with runs_lock:
        run = runs.get(run_id)
        if not run:
            return jsonify({"error": "Not found"}), 404
        return jsonify(run)


@app.route("/api/leaderboard")
def api_leaderboard():
    return jsonify(build_leaderboard())


@app.route("/api/results")
def api_results():
    return jsonify(load_results())


@app.route("/api/clear_results", methods=["POST"])
def api_clear():
    save_results([])
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n  Euler Solver Benchmark")
    print("  Open http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
