import os
import json
import tiktoken
from datasets import load_dataset

# 1. The Sycophancy & Flattery Filter (Hard Fail)
FLATTERY_PHRASES = [
    "great question", "thanks in advance", "happy to help", 
    "good luck", "brilliant", "awesome", "glad i could help",
    "you're a lifesaver", "excellent point", "thanks!", "thank you"
]

# 2. The Uncertainty & Hedging Filter (Hard Fail)
UNCERTAINTY_PHRASES = [
    "i could be wrong", "not an expert", "just a guess", 
    "i'm not sure", "tbh", "to be honest", "frankly", 
    "it might be", "perhaps", "i think maybe", "off the top of my head"
]

# 3. Pronoun limit for objectivity
FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

def filter_dialogue(text: str) -> bool:
    """Evaluates Q&A text for directness, removing flattery and uncertainty."""
    lower_text = text.lower()
    
    # Hard fail for conversational filler and flattery
    if any(phrase in lower_text for phrase in FLATTERY_PHRASES):
        return False
        
    # Hard fail for deceptive hedging or unconfident answers
    if any(phrase in lower_text for phrase in UNCERTAINTY_PHRASES):
        return False
        
    words = lower_text.split()
    if len(words) < 30: 
        return False # Drop fragments
        
    # Density fail for excessive first-person focus
    fp_count = sum(1 for word in words if word in FIRST_PERSON)
    density = fp_count / len(words)
    
    # Cap first-person usage at 2.0% to keep the dialogue strictly focused on the solution
    if density > 0.02:
        return False
        
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Target set to your 3 Billion requirement
    target_new_tokens = 3_000_000_000  
    
    output_file = "class_d_corpus.jsonl" 
    checkpoint_file = "checkpoint_class_d.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} Q&A threads.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    print("Connecting to Hugging Face to stream RedPajama-Data-1T (StackExchange subset)...")
    
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "stackexchange", split="train", streaming=True)
    
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
                
            if filter_dialogue(text):
                tokens = len(enc.encode(text, disallowed_special=()))
                tokens_this_session += tokens
                
                record = {
                    "source": "RedPajama_StackExchange",
                    "text": text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Progress update every 10 Million tokens
                if tokens_this_session % 10_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} Class D tokens...")
                
            # Save progress silently every 5,000 threads
            if raw_index % 5000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    # Final checkpoint save
    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"Dialogue extraction complete! Collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()