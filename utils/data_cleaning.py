import json
import re
import os

def clean_and_convert(input_path, output_path):
    print(f"Reading from {input_path}")
    
    # Read the entire file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # The file seems to have a mix of strict JSONL and some pretty-printed JSON arrays or trailing commas.
    # We will use regex to extract all "messages" arrays.
    
    # This regex looks for {"messages": [...]} regardless of newlines
    pattern = r'\{[\s]*"messages"[\s]*:[\s]*\[(.*?)\][\s]*\}'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    valid_conversations = []
    
    for match in matches:
        json_str = match.group(0)
        # Attempt to parse
        try:
            parsed = json.loads(json_str)
            if "messages" in parsed and len(parsed["messages"]) > 0:
                valid_conversations.append(parsed)
        except json.JSONDecodeError:
            # If it fails, try to fix common issues like trailing commas
            fixed_str = re.sub(r',\s*\}', '}', json_str)
            fixed_str = re.sub(r',\s*\]', ']', fixed_str)
            try:
                parsed = json.loads(fixed_str)
                if "messages" in parsed and len(parsed["messages"]) > 0:
                    valid_conversations.append(parsed)
            except json.JSONDecodeError as e:
                print(f"Failed to parse a block even after fixing. Error: {e}")
                print(f"Snippet: {json_str[:100]}...")

    print(f"Extracted {len(valid_conversations)} valid conversations.")
    
    # Write as strict JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in valid_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
            
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    input_file = "data/raw/dataset.json"
    output_file = "data/processed/cleaned_dataset.jsonl"
    clean_and_convert(input_file, output_file)
