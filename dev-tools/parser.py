import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_KEY'))


def read_role_prompt():
    # Get the directory of this script (dev-tools/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    role_file = os.path.join(script_dir, "roles", "parser.txt")

    with open(role_file, "r", encoding='utf-8') as file:
        content = file.read().strip()
        return content

def main():
    if len(sys.argv) < 2:
        print("Error: File name not provided.")
        return

    file_path = sys.argv[1]
    json_name = "strings" if len(sys.argv) < 3 else sys.argv[2]
    base_name = os.path.splitext(file_path)[0]

    # Check and create 'parsed' folder at project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parsed_folder = os.path.join(project_root, 'parsed')
    if not os.path.exists(parsed_folder):
        os.makedirs(parsed_folder)

    # Update file paths to save in 'parsed' folder
    parsed_file_path = os.path.join(parsed_folder, f"{os.path.basename(base_name)}.html")
    strings_file_path = os.path.join(parsed_folder, f"{json_name}.json")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Read existing data from JSON file
    if os.path.exists(strings_file_path):
        with open(strings_file_path, 'r', encoding='utf-8') as strings_file:
            existing_variables = json.load(strings_file)
    else:
        existing_variables = {}

    variables = existing_variables  # Start with existing variables

    # System message
    system_message = {"role": "system", "content": read_role_prompt()}

    try:
        with open(file_path, 'r', encoding='utf-8') as file, \
             open(parsed_file_path, 'w', encoding='utf-8') as parsed_file:

            for line in file:
                messages = [system_message, {"role": "user", "content": f"CODE: {line}"}]

                # Send request to GPT API for each line
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="gpt-4-0125-preview",
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )

                response = json.loads(chat_completion.choices[0].message.content)
                print(response)

                # Save 'code' content in parsed file
                code_content = response['code']
                parsed_file.write(code_content + '\n')

                # Update dictionary with variables and strings
                variable_name = response.get('variable')
                string_value = response.get('string')

                if variable_name and string_value:
                    variables[variable_name] = string_value

        # Write variable dictionary to JSON file
        with open(strings_file_path, 'w', encoding='utf-8') as strings_file:
            json.dump(variables, strings_file, indent=4)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
