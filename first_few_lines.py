import json

def inspect_data(file_path, num_lines=3):
    print(f"Reading the first {num_lines} articles from {file_path}...\n")
    print("-" * 50)
    
    # Open the file in read mode
    with open(file_path, 'r', encoding='utf-8') as f:
        for i in range(num_lines):
            # Read a single line
            line = f.readline()
            if not line:
                break
                
            # Convert the string back into a Python dictionary
            article = json.loads(line)
            
            # Print the data cleanly
            print(f"Title:  {article['title']}")
            print(f"URL:    {article['url']}")
            print(f"Tokens: {article['tokens']}")
            print(f"Text Preview:\n{article['text'][:500]}...\n") # Prints first 500 characters of text
            print("-" * 50)

if __name__ == "__main__":
    inspect_data("clean_non_manipulative_corpus.jsonl")