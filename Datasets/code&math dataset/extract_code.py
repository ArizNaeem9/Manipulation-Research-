import os
import json
import tiktoken
from datasets import load_dataset

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Target set to your 1.5 Billion split requirement
    target_new_tokens = 1_500_000_000  
    
    output_file = "class_c_corpus.jsonl" 
    checkpoint_file = "checkpoint_class_c.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} files.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    print("Connecting to Hugging Face to stream RedPajama-Data-1T (GitHub subset)...")
    
    # THE FIX: Switched to RedPajama's stable GitHub configuration
    # No trust_remote_code=True is needed because it uses standard file structures
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "github", split="train", streaming=True)
    
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget of {target_new_tokens:,} tokens reached. Stopping.")
                break
            
            text = item.get("text", "")
            if not text:
                continue
                
            # Structural Filter: Drop extremely short files (likely empty configs or stubs)
            if len(text.split()) < 50:
                continue
                
            tokens = len(enc.encode(text, disallowed_special=()))
            tokens_this_session += tokens
            
            record = {
                "source": "RedPajama_GitHub",
                "text": text,
                "tokens": tokens
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Progress update every 10 Million tokens
            if tokens_this_session % 10_000_000 < tokens:
                print(f"Collected {tokens_this_session:>,} Class C tokens...")
                
            # Save progress silently every 5,000 files
            if raw_index % 5000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    # Final checkpoint save
    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"Code extraction complete! Collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()