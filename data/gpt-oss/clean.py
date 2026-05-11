import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

files = sorted((SCRIPT_DIR / 'algorithm_pseudocode_tam').glob('*.txt'))

for file_path in files:
    try:
        # 1. Read the existing content
        content = file_path.read_text(encoding='utf-8')

        # 2. Extract the text between ''' and '''
        # Using re.DOTALL to capture multi-line content
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, content, flags=re.DOTALL)

        # 3. Join the matches (in case there are multiple)
        # We strip() each match to remove leading/trailing whitespace
        output_data = "\n\n".join([m.strip() for m in matches])

        # 4. Overwrite the file with the extracted content
        file_path.write_text(output_data, encoding='utf-8')
            
        print(f"Success! {file_path} has been updated with the extracted content.")

    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
