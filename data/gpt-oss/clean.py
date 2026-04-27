import re
import glob

Fs = glob.glob('*tam/*txt')

file_path = 'your_file.txt'  # Change this to your filename

for file_path in Fs:
    try:
        # 1. Read the existing content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. Extract the text between ''' and '''
        # Using re.DOTALL to capture multi-line content
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, content, flags=re.DOTALL)

        # 3. Join the matches (in case there are multiple)
        # We strip() each match to remove leading/trailing whitespace
        output_data = "\n\n".join([m.strip() for m in matches])

        # 4. Overwrite the file with the extracted content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output_data)
            
        print(f"Success! {file_path} has been updated with the extracted content.")

    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
