import requests
import pandas as pd
import openai
import os
from time import sleep

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def fetch_problem(problem_number):
    """Fetches a problem statement from Project Euler."""
    url = f"https://projecteuler.net/minimal={problem_number}"
    response = requests.get(url)
    return response.text.strip() if response.status_code == 200 else None

def solve_problem(problem):
    """Uses GPT to solve the given problem statement."""
    gpt_prompt = f"I send you a problem to solve.\n\n{problem}\n\nYour job is to output the numerical result and nothing else! If you can't solve it, output 0!"
    
    response = client.chat.completions.create(
        model='gpt-4o', 
        messages=[{"role": "user", "content": gpt_prompt}],
        max_tokens=10,
        temperature=0,
        timeout=30
    )
    
    return response.choices[0].message.content.strip()

def main():
    """Main function to process problems and compare solutions."""
    solutions = pd.read_csv('Solutions.md', sep='\s', header=None, names=['problem_nr', 'solution'])
    full_results = []
    
    for problem_nr, solution in solutions.values:
        problem = fetch_problem(problem_nr)
        if not problem:
            continue
        
        gpt_solution = solve_problem(problem)
        full_results.append([problem_nr, problem, solution, gpt_solution])
        
        if len(full_results) % 10 == 0:
            df_results = pd.DataFrame(full_results, columns=['problem_nr', 'problem', 'solution', 'gpt_solution'])
            df_results['success'] = df_results['solution'] == df_results['gpt_solution']
            df_results.to_csv('results.csv', index=False)
            print(f"Saved {len(full_results)} results.")
    
    # Calculate and print model accuracy
    df_results = pd.read_csv('results.csv')
    model_accuracy = df_results['success'].mean()
    print(f"Model Accuracy: {model_accuracy:.2%}")

if __name__ == "__main__":
    main()
