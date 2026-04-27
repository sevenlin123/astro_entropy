import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mxfp4Config
from tqdm import tqdm
import re

'''
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
quant_config = BitsAndBytesConfig(load_in_4bit=True)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config,
    dtype=torch.float16
)
'''
torch.cuda.empty_cache()
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
quant_config = Mxfp4Config(load_in_4bit=True, dequantize=False)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16, # Critical for Ampere
    device_map="auto",
    #max_memory={0: "8GiB", "cpu": "32GiB"},
    attn_implementation="eager"
)


def clean_stage1_algorithm(raw_text, original_prompt=""):
    """
    Cleans Stage 1 outputs by removing the prompt, DeepSeek reasoning tags, 
    and common conversational filler before embedding.
    """
    # 1. Strip the exact original prompt if it's concatenated in the output
    if original_prompt and original_prompt in raw_text:
        raw_text = raw_text.replace(original_prompt, "")
        
    # 2. Erase DeepSeek-R1 reasoning blocks (<think>...</think>)
    # re.DOTALL ensures the regex matches across multiple newlines
    cleaned_text = re.sub(r'^.*?\</think>', '', raw_text, flags=re.DOTALL)
    
    # 3. Remove common conversational boilerplate
    filler_patterns = [
        r"^(Here is the algorithm.*?:?\n)",
        r"^(Sure, here are the computational steps.*?:?\n)",
        r"^(Certainly!.*?\n)"
    ]
    
    cleaned_text = cleaned_text.strip()
    for pattern in filler_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        
    return cleaned_text.strip()

def extract_stage2_code(raw_text):
    """
    Extracts purely the Python code from Stage 2 outputs, ignoring all surrounding text.
    """
    # Match everything explicitly inside ```python ... ```
    match = re.search(r'```python(.*?)```', raw_text, flags=re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: if the model just used ``` without specifying 'python'
    match_generic = re.search(r'```(.*?)```', raw_text, flags=re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
        
    # Return empty string if the generation truncated before formatting
    return ""



def generate(prompt, temperature = 0.85):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            min_new_tokens=300,
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
            repetition_penalty=1.05
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_algorithm(paper_text):

    prompt = f"""
You are analyzing a computational astrophysics method.

Based on the following Paper description, extract a plausible
computational algorithm that could implement the core reconstruction framework.
Provide the algorithm in pseudocode with explicit computational steps.

Focus only on the core computational method.

Important simplifications:
- Ignore validation procedures (LOOCV, cross validation, bootstrapping)
- Ignore plotting or visualization


Important rules:
- Do NOT show reasoning or thinking.
- Do NOT explain the answer.
- Do NOT write code.
- Only return the final structured algorithm description.
- Only describe the algorithm steps.
- Only return the final requested output.

Paper:
{paper_text}

"""
    temp = 0.75

    return generate(prompt, temperature=temp)


def generate_code(algorithm):

    prompt = f"""
Write a minimal Python implementation of the following algorithm.

Requirements:

- implement only the core reconstruction framework
- ignore validation methods (LOOCV, cross-validation)
- Do not need plotting
- generate synthetic example input data inside the script
- Do NOT show reasoning or thinking.
- Do NOT explain the code.
- Return only the Python code.

The code should:

1. define a spectral model
2. generate synthetic spectra
2. generate photometric data from synthetic spectra
3. reconstruct a synthetic spectrum from photometruic

Constraints:

- use numpy, scipy, and scikit-learn
- produce runnable Python code
- define clear functions

Return only the code.
"""

    #temp = random.uniform(0.6, 1.0)
    temp = 0.75

    return generate(prompt, temperature=temp)


def main():

    os.makedirs("algorithm", exist_ok=True)
    os.makedirs("codes", exist_ok=True)

    N = 200
    types = ['tam']
    paper_file = {'t':"paper/paper_T.txt", 'ta': "paper/paper_TA.txt", 'tam':"paper/paper.txt"}
    for t in types:
        paper = open(paper_file[t]).read()
        for i in tqdm(range(N)):
            print("Extracting algorithm...")
            algorithm = extract_algorithm(paper)
            clear_algorithm = clean_stage1_algorithm(algorithm)
            with open(f"algorithm_pseudocode_{t}/algorithm_{i:04d}.txt","w") as f:
                f.write(clear_algorithm)

            #print("Generating codes...")
            #code = generate_code(clear_algorithm)
            #clear_code =  extract_stage2_code(code)

            #with open(f"codes/code_{i:04d}.py","w") as f:
            #    f.write(clear_code)


if __name__ == "__main__":
    main()
