import os
import json
import tiktoken
from datasets import load_dataset

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Target set for the remaining 1.5 Billion tokens
    target_new_tokens = 3_000_000_000  
    
    output_file = "class_c_corpus.jsonl" 
    checkpoint_file = "checkpoint_math.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} math documents.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    print("Connecting to Hugging Face to stream open-web-math/open-web-math...")
    
    # This dataset perfectly preserves LaTeX and structured mathematical reasoning
    dataset = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget of {target_new_tokens:,} math tokens reached. Stopping.")
                break
            
            text = item.get("text", "")
            if not text:
                continue
                
            # Structural Filter: Drop short fragments to ensure we capture full proofs/papers
            if len(text.split()) < 100:
                continue
                
            # Direct encoding; pure math/logic does not require subjectivity filtering
            tokens = len(enc.encode(text, disallowed_special=()))
            tokens_this_session += tokens
            
            record = {
                "source": "OpenWebMath",
                "text": text,
                "tokens": tokens
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Progress update every 10 Million tokens
            if tokens_this_session % 10_000_000 < tokens:
                print(f"Collected {tokens_this_session:>,} Math tokens...")
                
            # Save progress silently every 2,000 files
            if raw_index % 2000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    # Final checkpoint save
    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"Math extraction complete! Collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()