import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

def read_role_prompt(language_code):
    # Get the directory of this script (dev-tools/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    role_file = os.path.join(script_dir, "roles", "strings-translator.txt")

    with open(role_file, "r", encoding='utf-8') as file:
        content = file.read().strip()
        return content.replace("{{language}}", language_code)

def get_language_code(filename):
    # Extract language code from filename, assuming it's before '.json'
    return os.path.splitext(filename)[0].split('_')[-1]

def main():
    if len(sys.argv) != 3:
        print("Error: incorrect number of arguments provided.")
        print("Usage: python strings-translator.py <input_json_file> <output_json_file>")
        return

    input_json_filename = sys.argv[1]
    output_json_filename = sys.argv[2]

    # Build complete path for input and output files within templates/lang
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_lang_path = os.path.join(project_root, 'templates', 'lang')
    input_json_path = os.path.join(templates_lang_path, input_json_filename)
    output_json_path = os.path.join(templates_lang_path, output_json_filename)

    # Main change: Use output file language code for translation
    target_language_code = get_language_code(output_json_filename)

    if not os.path.exists(input_json_path):
        print(f"Error: File '{input_json_path}' not found.")
        return

    existing_translations = {}

    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as existing_file:
            existing_translations = json.load(existing_file)

    translations = {}

    system_message_content = read_role_prompt(target_language_code)
    system_message = {"role": "system", "content": system_message_content}

    try:
        with open(input_json_path, 'r', encoding='utf-8') as file:
            content = json.load(file)

        for variable, string in content.items():
            if variable not in existing_translations:
                line = f'LINE: "{variable}": "{string}"'
                print(line)
                messages = [system_message, {"role": "user", "content": line}]

                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="gpt-4-0125-preview",
                    temperature=0.9,
                    response_format={"type": "json_object"}
                )

                response = json.loads(chat_completion.choices[0].message.content)

                variable_name = response.get('variable')
                translated_string = response.get('string')

                print(f"Response, variable: {variable_name} , string: {translated_string}")

                if variable_name and translated_string:
                    translations[variable_name] = translated_string

        # Update existing dictionary with new translations
        existing_translations.update(translations)

        with open(output_json_path, 'w', encoding='utf-8') as output_file:
            json.dump(existing_translations, output_file, indent=4)

        print("Job Done!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()