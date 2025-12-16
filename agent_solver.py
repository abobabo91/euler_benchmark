import requests
import pandas as pd
import openai
import anthropic
import google.generativeai as genai
import os
import re
import sys
import argparse
import subprocess
import time
import json
import traceback

# --- Configuration & Setup ---

SECRETS_PATH = r"C:\Users\abele\.streamlit\secrets.toml"
WORKSPACE_DIR = "agent_workspace"

def load_secrets(filepath):
    """Parses a simple TOML file to extract API keys."""
    secrets = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        patterns = {
            'OPENAI_API_KEY': r'OPENAI_API_KEY\s*=\s*"([^"]+)"',
            'GEMINI_API_KEY': r'GEMINI_API_KEY\s*=\s*"([^"]+)"',
            'ANTHROPIC_API_KEY': r'ANTHROPIC_API_KEY\s*=\s*"([^"]+)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                secrets[key] = match.group(1)
                
    except FileNotFoundError:
        print(f"Error: Secrets file not found at {filepath}")
    except Exception as e:
        print(f"Error reading secrets: {e}")
        
    return secrets

# Load keys
secrets = load_secrets(SECRETS_PATH)
OPENAI_API_KEY = secrets.get('OPENAI_API_KEY')
GEMINI_API_KEY = secrets.get('GEMINI_API_KEY')
ANTHROPIC_API_KEY = secrets.get('ANTHROPIC_API_KEY')

# Create workspace for agent files if it doesn't exist
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# --- Tool Definitions ---

def write_file(filename, content):
    """Writes content to a file in the workspace."""
    filepath = os.path.join(WORKSPACE_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing file: {e}"

def read_file(filename):
    """Reads content from a file in the workspace."""
    filepath = os.path.join(WORKSPACE_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def run_python(filename):
    """Executes a python script from the workspace and captures stdout."""
    filepath = os.path.join(WORKSPACE_DIR, filename)
    if not os.path.exists(filepath):
        return f"Error: File {filename} does not exist."

    try:
        # Run with a timeout to prevent infinite loops
        # Since cwd is set to WORKSPACE_DIR, we just pass the filename, not the full path
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=WORKSPACE_DIR  # Execute in workspace directory
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        
        if not output.strip():
            return "[No Output]"
            
        return output
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out (10s limit)."
    except Exception as e:
        return f"Error executing script: {e}"

# --- Agent Logic ---

SYSTEM_PROMPT = """
You are an autonomous coding agent solving Project Euler problems.
You have access to a local workspace and tools to write and run Python code.

TOOLS:
1. write_file(filename, content): Create or overwrite a Python script.
2. run_python(filename): Execute a Python script and see the output.
3. read_file(filename): Read a file to check its content.

PROTOCOL:
To use a tool, you MUST format your response strictly as a JSON block.
Do not output anything else outside the JSON block when using a tool.

Example - Writing a file:
```json
{
    "tool": "write_file",
    "filename": "solution.py",
    "content": "print(1+1)"
}
```

Example - Running a file:
```json
{
    "tool": "run_python",
    "filename": "solution.py"
}
```

Example - Final Answer:
If you have found the solution, output a JSON with "final_answer".
```json
{
    "final_answer": "12345"
}
```

IMPORTANT:
- Always verify your solution by running the code.
- If the code errors, read the error, fix the code, and try again.
- The problem asks for a single number as the solution.
- Output ONLY valid JSON.
"""

class Agent:
    def __init__(self, provider, model_name, api_key):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.history = []
        self.max_steps = 10

    def chat_openai(self, messages):
        client = openai.OpenAI(api_key=self.api_key)
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {e}"

    def chat_anthropic(self, messages):
        client = anthropic.Anthropic(api_key=self.api_key)
        # Convert standard messages to Anthropic format if needed, but basic structure is similar
        # Anthropic doesn't support 'system' role in messages list in older versions, 
        # but modern API uses top-level system param.
        
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
        user_assistant_msgs = [m for m in messages if m['role'] != 'system']
        
        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0,
                system=system_msg,
                messages=user_assistant_msgs
            )
            return response.content[0].text
        except Exception as e:
            return f"API Error: {e}"

    def chat_gemini(self, messages):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        
        # Gemini history format conversion
        gemini_hist = []
        system_instruction = next((m['content'] for m in messages if m['role'] == 'system'), "")
        
        # We'll treat system prompt as part of the first user message for Gemini 
        # or use system_instruction if supported (depends on version).
        # Safe fallback: Prepend system prompt to first user message.
        
        for m in messages:
            if m['role'] == 'user':
                role = 'user'
                content = m['content']
            elif m['role'] == 'assistant':
                role = 'model'
                content = m['content']
            elif m['role'] == 'system':
                continue # Handled separately
            else:
                continue
            gemini_hist.append({'role': role, 'parts': [content]})
            
        if gemini_hist and messages[0]['role'] == 'system':
             gemini_hist[0]['parts'][0] = f"{messages[0]['content']}\n\n{gemini_hist[0]['parts'][0]}"

        try:
            # Generate content for the last turn
            # Note: generative-ai python lib structure is a bit different, 
            # usually you build a chat session or just generate content.
            # For simplicity, we'll just send the whole history as prompt context if chat structure is complex
            # But let's try standard chat.
            
            chat = model.start_chat(history=gemini_hist[:-1])
            response = chat.send_message(gemini_hist[-1]['parts'][0], generation_config={'temperature': 0})
            return response.text
        except Exception as e:
            return f"API Error: {e}"

    def solve(self, problem):
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Solve this problem: {problem}"}
        ]
        
        print(f"Agent starts solving... (Model: {self.model_name})")

        for step in range(self.max_steps):
            # 1. Get Model Response
            response_text = ""
            if self.provider == 'openai':
                response_text = self.chat_openai(self.history)
            elif self.provider == 'claude':
                response_text = self.chat_anthropic(self.history)
            elif self.provider == 'gemini':
                response_text = self.chat_gemini(self.history)
            
            # Print brief log
            print(f"Step {step+1}: Model returned {len(response_text)} chars")
            
            self.history.append({"role": "assistant", "content": response_text})

            # 2. Parse Tool Call
            try:
                # Find JSON block
                json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
                if not json_match:
                     # Try searching for raw JSON if code blocks are missing
                    json_match = re.search(r'({.*"tool".*})', response_text, re.DOTALL)
                
                if json_match:
                    tool_data = json.loads(json_match.group(1))
                    
                    if "final_answer" in tool_data:
                        return str(tool_data["final_answer"])
                    
                    tool_name = tool_data.get("tool")
                    tool_output = ""
                    
                    print(f"  -> Executing Tool: {tool_name}")
                    
                    if tool_name == "write_file":
                        tool_output = write_file(tool_data["filename"], tool_data["content"])
                    elif tool_name == "run_python":
                        tool_output = run_python(tool_data["filename"])
                    elif tool_name == "read_file":
                        tool_output = read_file(tool_data["filename"])
                    else:
                        tool_output = f"Unknown tool: {tool_name}"
                    
                    # 3. Feed Output back to history
                    user_msg = f"Tool Output ({tool_name}):\n{tool_output}"
                    self.history.append({"role": "user", "content": user_msg})
                    
                else:
                    # No tool call found, maybe it's just talking?
                    # If it looks like a final answer but not JSON
                    if "final answer" in response_text.lower():
                        # Try to extract number? Or just ask to format?
                        self.history.append({"role": "user", "content": "Please output the answer strictly in the JSON format: {'final_answer': 'YOUR_ANSWER'}"})
                    else:
                        self.history.append({"role": "user", "content": "I did not find a valid JSON tool call. Please check your formatting."})
                        
            except json.JSONDecodeError:
                self.history.append({"role": "user", "content": "Invalid JSON format. Please try again."})
            except Exception as e:
                self.history.append({"role": "user", "content": f"System Error: {e}"})

        return "Failed to find solution in max steps."

# --- Main Logic ---

MODEL_CONFIG = {
    # OpenAI Models
    'gpt-4o-mini': ('openai', 'gpt-4o-mini'),
    'gpt-4o': ('openai', 'gpt-4o'),
    
    # Gemini Models
    'gemini-1.5-flash': ('gemini', 'gemini-1.5-flash'),
    'gemini-1.5-pro': ('gemini', 'gemini-1.5-pro'),
    
    # Claude Models
    'claude-3-5-sonnet': ('claude', 'claude-3-5-sonnet-20241022'),
    'claude-3-haiku': ('claude', 'claude-3-haiku-20240307'),
}

def fetch_problem(problem_number):
    try:
        problem_number = int(problem_number)
    except ValueError:
        return None

    url = f"https://projecteuler.net/minimal={problem_number}"
    try:
        response = requests.get(url)
        return response.text.strip() if response.status_code == 200 else None
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Agentic Euler Solver")
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIG.keys()), help='The specific AI model to use')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of problems to solve')
    args = parser.parse_args()

    choice_key = args.model

    if not choice_key:
        print("Select AI Model for Agent:")
        sorted_keys = sorted(MODEL_CONFIG.keys())
        for idx, key in enumerate(sorted_keys, 1):
            provider, model_name = MODEL_CONFIG[key]
            print(f"{idx}. {key} ({provider}: {model_name})")
            
        choice = input(f"Enter choice (1-{len(sorted_keys)}): ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sorted_keys):
                choice_key = sorted_keys[choice_idx]
            else:
                print("Invalid choice.")
                return
        except ValueError:
            print("Invalid input.")
            return

    provider, api_model_name = MODEL_CONFIG[choice_key]
    
    # Select Key
    api_key = None
    if provider == 'openai': api_key = OPENAI_API_KEY
    elif provider == 'gemini': api_key = GEMINI_API_KEY
    elif provider == 'claude': api_key = ANTHROPIC_API_KEY
    
    if not api_key:
        print(f"Error: No API key found for {provider}")
        return

    agent = Agent(provider, api_model_name, api_key)

    try:
        solutions = pd.read_csv('Solutions.md', sep='\s+', header=None, names=['problem_nr', 'solution'])
    except Exception as e:
        print(f"Error reading Solutions.md: {e}")
        return

    full_results = []
    
    # Interactive single problem mode or batch?
    # Let's do batch like solver.py but maybe just first 5 for testing?
    # User can ctrl+c to stop.
    
    print(f"Starting Agentic Solver with {choice_key}...")
    
    count = 0
    for problem_nr, solution in solutions.values:
        if args.limit and count >= args.limit:
            break
            
        if pd.isna(solution): continue
            
        problem_text = fetch_problem(problem_nr)
        if not problem_text: continue
        
        print(f"\n=== Solving Problem {int(problem_nr)} ===")
        
        start_time = time.time()
        agent_solution = agent.solve(problem_text)
        duration = time.time() - start_time
        
        success = str(solution).strip() == str(agent_solution).strip()
        print(f"Result: {agent_solution} (Expected: {solution}) | Success: {success} | Time: {duration:.2f}s")
        
        full_results.append([problem_nr, agent_solution, success, duration])
        
        # Save progress
        df = pd.DataFrame(full_results, columns=['problem_nr', 'agent_solution', 'success', 'duration'])
        df.to_csv('agent_results.csv', index=False)
        count += 1

if __name__ == "__main__":
    main()
