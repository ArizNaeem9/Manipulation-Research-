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
    """Returns False if it contains manipulative or heavily subjective language."""
    if BIAS_PATTERN.search(text):
        return False
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    # We need 2 Billion tokens to reach your 6B total
    target_tokens = 2_000_000_000  
    current_tokens = 0
    output_file = "common_corpus_non_fiction.jsonl"
    
    print("Connecting to Hugging Face to stream PleIAs/common_corpus...")
    
    # 2. Stream the dataset
    # We use streaming=True because this dataset is roughly 2 trillion tokens in total
    dataset = load_dataset("PleIAs/common_corpus", split="train", streaming=True)
    
    # Define the strict non-fiction collections we want to pull from
    TARGET_CATEGORIES = {"Open Science", "Open Government"}
    
    print(f"Extracting strict non-fiction text to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if current_tokens >= target_tokens:
                print(f"\nTarget of {target_tokens:,} tokens reached. Stopping extraction.")
                break
            
            # 3. Filter for specific domains (Science and Government)
            open_type = item.get("open_type", "")
            if open_type not in TARGET_CATEGORIES:
                continue
                
            text = item.get("text", "")
            if not text:
                continue
                
            paragraphs = text.split("\n")
            clean_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if len(para) < 50: 
                    continue
                    
                # 4. Apply the neutrality filter
                if is_objective(para):
                    clean_paragraphs.append(para)
            
            if clean_paragraphs:
                clean_text = "\n".join(clean_paragraphs)
                
                # Count tokens
                tokens = len(enc.encode(clean_text, disallowed_special=()))
                current_tokens += tokens
                
                # Save the clean data incrementally
                record = {
                    "title": item.get("title", "Untitled"),
                    "category": open_type,
                    "text": clean_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Terminal progress update
                if current_tokens % 10_000_000 < tokens:
                    print(f"Collected {current_tokens:>,} / {target_tokens:>,} tokens...")

    print("Common Corpus data collection pipeline complete!")

if __name__ == "__main__":
    main()