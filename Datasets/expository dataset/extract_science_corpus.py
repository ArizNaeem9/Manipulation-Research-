import os
import json
import re
import tiktoken
from datasets import load_dataset

# 1. Define the Subjectivity & Manipulation Lexicon
SUBJECTIVE_WORDS = {
    "obviously", "clearly", "undoubtedly", "sadly", "fortunately", 
    "unfortunately", "surprisingly", "shockingly", "terrible", 
    "horrible", "amazing", "incredible", "best", "worst", "stupid",
    "genius", "ridiculous", "outrageous", "disgusting", "pathetic",
    "brilliant", "gorgeous", "ugly", "evil", "heroic", "masterpiece",
    "tragic", "wonderful", "awful", "fantastic", "dreadful"
}

BIAS_PATTERN = re.compile(r'\b(?:' + '|'.join(SUBJECTIVE_WORDS) + r')\b', re.IGNORECASE)

def is_objective(text: str) -> bool:
    """Returns False if it contains manipulative language."""
    if BIAS_PATTERN.search(text):
        return False
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Set this to how many NEW tokens you want to collect during this specific run
    target_new_tokens = 10_000_000_000  
    
    output_file = "clean_science_corpus.jsonl"
    checkpoint_file = "checkpoint_science.json"
    
    # 2. Checkpoint Logic: Load previous progress to avoid duplicates
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming from checkpoint! Skipping the first {start_index:,} raw articles.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    print("Connecting to Hugging Face to stream allenai/peS2o...")
    dataset = load_dataset("allenai/peS2o", "v2", split="train", streaming=True)
    
    # Skip the articles we already processed in previous runs
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    print(f"Extracting objective text and APPENDING to {output_file}...")
    
    # 3. Use "a" (append) mode instead of "w" (overwrite)
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1 # Track our exact position in the Hugging Face stream
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget of {target_new_tokens:,} NEW tokens reached. Stopping.")
                break
                
            text = item.get("text", "")
            if not text:
                continue
                
            paragraphs = text.split("\n")
            clean_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if len(para) < 50: 
                    continue
                if is_objective(para):
                    clean_paragraphs.append(para)
            
            if clean_paragraphs:
                clean_text = "\n".join(clean_paragraphs)
                tokens = len(enc.encode(clean_text, disallowed_special=()))
                tokens_this_session += tokens
                
                record = {
                    "id": item.get("id", "Unknown_ID"),
                    "source": item.get("source", "Unknown_Source"),
                    "text": clean_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if tokens_this_session % 10_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} / {target_new_tokens:>,} NEW tokens...")
            
            # 4. Save progress silently every 10,000 documents
            # If you Ctrl+C to quit early, it remembers where you left off
            if raw_index % 10000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    # Final checkpoint save when the target is reached
    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print("Data collection pipeline complete! State saved.")

if __name__ == "__main__":
    main()