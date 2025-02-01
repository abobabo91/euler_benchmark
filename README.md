# Project Euler Solver LLM Benchmark

This is a simple project that tries to solve **Project Euler** problems using **OpenAI's GPT API**. The script gets the problems, asks GPT to solve them, and checks if the answers match the correct ones. It keeps track of everything in a CSV file.

## What It Does
- Pulls problems from Project Euler
- Asks GPT-4o to solve them
- Compares the answers with real solutions
- Saves the results for reference

## What is Project Euler?
[Project Euler](https://projecteuler.net/) is a collection of tricky math and programming problems. These aren’t just basic math questions—they often need smart algorithms and problem-solving skills. Because of this, they're a great way to see how good an AI model really is at thinking through tough problems.

## Testing AI with These Problems
Since Project Euler problems require actual reasoning and not just memorization, they make a great way to test an AI’s problem-solving ability. That said, it’s possible that some answers are already in the AI’s training data, which could affect the results.

## GPT-4o Results
results.csv is the results for the current 'gpt-4o' model.

## What You Need
- Python 3.x
- `requests`
- `pandas`
- `openai`

## How to Set It Up
git clone https://github.com/YOUR_USERNAME/ProjectEulerSolver.git
cd ProjectEulerSolver
pip install -r requirements.txt

## How to Use It
1. **Set Up Your API Key**
   - Add your OpenAI API key to the code
2. **Run the Script**
   python solver.py
   ```

The results will be saved in `results.csv`.

## License
This project is open-source under the MIT License. Check `LICENSE` for details.
