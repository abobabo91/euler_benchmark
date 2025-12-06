import requests
import pandas as pd
import openai
import anthropic
import google.generativeai as genai
import os
import re
import sys
import argparse

# --- Configuration & Setup ---

SECRETS_PATH = r"C:\Users\abele\.streamlit\secrets.toml"

def load_secrets(filepath):
    """Parses a simple TOML file to extract API keys."""
    secrets = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Regex to find keys. Handles potential variations in whitespace and quoting.
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

# --- Model Definitions ---

# Maps CLI/User choice to (Provider, API Model Name)
MODEL_CONFIG = {
    # OpenAI Models
    'gpt-5.1': ('openai', 'gpt-5.1'),
    'gpt-5-mini': ('openai', 'gpt-5-mini'),
    'gpt-5-nano': ('openai', 'gpt-5-nano'),
    'gpt-5-pro': ('openai', 'gpt-5-pro'),
    'gpt-5': ('openai', 'gpt-5'),
    'o1': ('openai', 'o1'),
    'gpt-4o-mini': ('openai', 'gpt-4o-mini'),
    
    # Gemini Models
    'gemini-3-pro': ('gemini', 'gemini-3-pro-preview'),
    'gemini-2.5-flash': ('gemini', 'gemini-2.5-flash'),
    'gemini-2.5-flash-lite': ('gemini', 'gemini-2.5-flash-lite'),
    'gemini-2.5-pro': ('gemini', 'gemini-2.5-pro'),
    
    # Claude Models
    'claude-sonnet-4.5': ('claude', 'claude-sonnet-4-5-20250929'),
    'claude-haiku-4.5': ('claude', 'claude-haiku-4-5-20251001'),
    'claude-opus-4.5': ('claude', 'claude-opus-4-5-20251101'),
}

# --- Solver Functions ---

def solve_with_openai(problem, api_key, model_name):
    if not api_key:
        return "Error: Missing OpenAI API Key"
    
    client = openai.OpenAI(api_key=api_key)
    prompt = f"I send you a problem to solve.\n\n{problem}\n\nYour job is to output the numerical result and nothing else! If you can't solve it, output 0!"
    
    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20, 
            temperature=0,
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def solve_with_gemini(problem, api_key, model_name):
    if not api_key:
        return "Error: Missing Gemini API Key"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"I send you a problem to solve.\n\n{problem}\n\nYour job is to output the numerical result and nothing else! If you can't solve it, output 0!"
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=20,
                temperature=0
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def solve_with_claude(problem, api_key, model_name):
    if not api_key:
        return "Error: Missing Anthropic API Key"
    
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"I send you a problem to solve.\n\n{problem}\n\nYour job is to output the numerical result and nothing else! If you can't solve it, output 0!"
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=20,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error: {e}"

# --- Main Logic ---

def fetch_problem(problem_number):
    """Fetches a problem statement from Project Euler."""
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
    parser = argparse.ArgumentParser(description="Project Euler Solver Benchmark")
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIG.keys()), help='The specific AI model to use')
    args = parser.parse_args()

    choice_key = args.model

    # Interactive selection if not provided via CLI
    if not choice_key:
        print("Select AI Model:")
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
    print(f"Using model: {choice_key} -> {api_model_name} (Provider: {provider})")

    try:
        solutions = pd.read_csv('Solutions.md', sep='\s+', header=None, names=['problem_nr', 'solution'])
    except Exception as e:
        print(f"Error reading Solutions.md: {e}")
        return

    full_results = []
    
    for problem_nr, solution in solutions.values:
        if pd.isna(solution):
            print(f"Skipping Problem {problem_nr}: No solution provided.")
            continue
            
        problem_text = fetch_problem(problem_nr)
        if not problem_text:
            print(f"Skipping Problem {problem_nr}: Could not fetch problem.")
            continue
        
        print(f"Solving Problem {int(problem_nr)}...")
        
        gpt_solution = ""
        if provider == 'openai':
            gpt_solution = solve_with_openai(problem_text, OPENAI_API_KEY, api_model_name)
        elif provider == 'gemini':
            gpt_solution = solve_with_gemini(problem_text, GEMINI_API_KEY, api_model_name)
        elif provider == 'claude':
            gpt_solution = solve_with_claude(problem_text, ANTHROPIC_API_KEY, api_model_name)
        
        success = str(solution).strip() == str(gpt_solution).strip()
        
        full_results.append([problem_nr, problem_text, solution, gpt_solution, success])
        
        if len(full_results) % 10 == 0:
            df_results = pd.DataFrame(full_results, columns=['problem_nr', 'problem', 'solution', 'model_solution', 'success'])
            df_results.to_csv('results.csv', index=False)
            print(f"Saved {len(full_results)} results.")
    
    if full_results:
        df_results = pd.DataFrame(full_results, columns=['problem_nr', 'problem', 'solution', 'model_solution', 'success'])
        df_results.to_csv('results.csv', index=False)
        model_accuracy = df_results['success'].mean()
        print(f"Model Accuracy ({choice_key}): {model_accuracy:.2%}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
