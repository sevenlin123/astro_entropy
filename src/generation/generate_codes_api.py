import os
import random
from tqdm import tqdm
from google import genai
from google.genai import types

# Initialize the Gemini client
# Note: Ensure you have your GEMINI_API_KEY set in your environment variables
client = genai.Client()

# gemini-2.5-flash is extremely fast. If you need more complex reasoning 
# for the astrophysics logic, you can change this to "gemini-2.5-pro"
MODEL_NAME = "gemini-2.5-flash" 

def generate(prompt, temperature=0.85):
    """Generates a response using the Gemini API."""
    
    # Map your previous generation parameters to Gemini's config
    config = types.GenerateContentConfig(
        temperature=temperature, # This now properly uses the passed argument!
        top_p=0.95,
        max_output_tokens=8192,
        # Note: 'min_new_tokens' and 'repetition_penalty' do not have exact 
        # 1:1 equivalents in the standard Gemini API, so they are omitted. 
        # Gemini generally handles repetition well without the penalty.
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config
    )

    return response.text


def extract_algorithm(paper_text):

    prompt = f"""
You are an astrophysics research assistant.

Extract the computational algorithm required to reproduce the method.

DO NOT WRITE CODE.
Do NOT include reasoning or explanation.
Return only the final result.

Paper:
{paper_text}

Return structured algorithm steps.
"""

    return generate(prompt, temperature=0.2) # Lowered temp slightly for extraction accuracy, optional!


def generate_code(algorithm):

    prompt = f"""
You are a scientific Python programmer.

Implement the following algorithm in Python.


Requirements:
- generate 50 synthetic spectra 
- mock the photometry from synthetic spectrum
- apply the Implement to reconstruct spectrum with mock photometry
- plot mock photometry, reconstruct spectrum, and compare to synthetic spectrum
- runnable code
- numpy/scipy allowed
- include functions

Algorithm:
{algorithm}

Return ONLY Python code.
"""

    # Randomize temperature between 0.6 and 1.0 to get diverse code solutions
    temp = random.uniform(0.6, 1.0)

    return generate(prompt, temperature=temp)


def main():

    # Make sure paper/paper.txt exists before running!
    paper = open("paper/paper.txt").read()

    os.makedirs("algorithm", exist_ok=True)
    os.makedirs("codes", exist_ok=True)

    N = 20

    for i in tqdm(range(N)):
        print(f"\nIteration {i+1}/{N}")
        
        print("Extracting algorithm...")
        algorithm = extract_algorithm(paper)
        with open(f"algorithm/algorithm_{i:04d}.txt", "w") as f:
            f.write(algorithm)

        print("Generating codes...")
        code = generate_code(algorithm)
        with open(f"codes/code_{i:04d}.py", "w") as f:
            # Strip out markdown formatting if the model wraps it in ```python ... ```
            clean_code = code.replace("```python\n", "").replace("```", "")
            f.write(clean_code)


if __name__ == "__main__":
    main()