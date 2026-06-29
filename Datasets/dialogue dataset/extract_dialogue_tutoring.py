import os
import json
import tiktoken
from datasets import load_dataset

# 1. The Sycophancy & Flattery Filter (Hard Fail)
FLATTERY_PHRASES = [
    "great question", "happy to help", "good luck", "brilliant", 
    "awesome", "glad i could help", "you're a lifesaver", "thanks!"
]

# 2. The Uncertainty & Hedging Filter (Hard Fail)
UNCERTAINTY_PHRASES = [
    "i could be wrong", "not an expert", "just a guess", 
    "i'm not sure", "tbh", "to be honest", "frankly", 
    "it might be", "perhaps", "i think maybe", "off the top of my head"
]

# 3. The Condescension Filter (Hard Fail)
CONDESCENDING_PHRASES = [
    "obviously", "it's basic", "as i already said", "you should know",
    "it's simple really", "it's completely wrong", "read a book",
    "google it", "do your own research", "that's a stupid question"
]

def filter_answer(text: str) -> bool:
    """Evaluates a tutoring answer for directness, removing flattery, hedging, and arrogance."""
    lower_text = text.lower()
    
    # Check 1: Drop conversational filler and flattery
    if any(phrase in lower_text for phrase in FLATTERY_PHRASES):
        return False
        
    # Check 2: Drop unconfident, deceptive, or guessing answers
    if any(phrase in lower_text for phrase in UNCERTAINTY_PHRASES):
        return False
        
    # Check 3: Drop argumentative or condescending explanations
    if any(phrase in lower_text for phrase in CONDESCENDING_PHRASES):
        return False
        
    # Drop fragments (a good explanation requires depth)
    words = lower_text.split()
    if len(words) < 40: 
        return False 
        
    # First-person limit (Cap at 2.0% to keep focus on the subject, not the tutor)
    fp_count = sum(1 for word in words if word in {"i", "me", "my", "mine", "myself"})
    if (fp_count / len(words)) > 0.02:
        return False
        
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    target_new_tokens = 2_000_000_000  
    
    output_file = "class_d_tutoring.jsonl" 
    checkpoint_file = "checkpoint_tutoring.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} questions.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    # THE FIX: Pointing to the stable, modernized Parquet version
    print("Connecting to Hugging Face to stream sentence-transformers/eli5...")
    dataset = load_dataset("sentence-transformers/eli5", split="train", streaming=True)
    
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget reached. Stopping.")
                break
            
            # The sentence-transformers format directly provides 'question' and 'answer'
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not question or not answer:
                continue
                
            if filter_answer(answer):
                # Format as a clean Q&A dialogue block
                dialogue_text = f"Question: {question}\n\nAnswer: {answer}"
                
                tokens = len(enc.encode(dialogue_text, disallowed_special=()))
                tokens_this_session += tokens
                
                record = {
                    "source": "ELI5_Tutoring",
                    "text": dialogue_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if tokens_this_session % 5_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} Class D tutoring tokens...")
                
            if raw_index % 2000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"Tutoring extraction complete! Collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()