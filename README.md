# Project Euler Solver

This project automates solving **Project Euler** problems using **OpenAI's GPT API**. It fetches problem statements, attempts solutions using GPT, and compares results against known solutions.

## Features
- Fetches problems from Project Euler
- Uses GPT-4o to solve the problems
- Compares GPT-generated answers with known solutions
- Logs results in a CSV file

## Requirements
- Python 3.x
- `requests`
- `pandas`
- `openai`

## Installation

Clone the repository:
```sh
git clone https://github.com/YOUR_USERNAME/ProjectEulerSolver.git
cd ProjectEulerSolver
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

1. **Set up API Key**
   - Rename `.env.example` to `.env` and add your OpenAI API key.

2. **Run the script**
   ```sh
   python solver.py
   ```

Results will be saved in `results.csv`.

## License
MIT License. See `LICENSE` for details.

