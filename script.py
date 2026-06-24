import json
import re
import tiktoken
from datasets import load_dataset

# 1. Define the Subjectivity & Manipulation Lexicon
# This list filters out emotional, persuasive, or highly subjective words 
# that violate pure, objective non-fiction reporting.
SUBJECTIVE_WORDS = {
    "obviously", "clearly", "undoubtedly", "sadly", "fortunately", 
    "unfortunately", "surprisingly", "shockingly", "terrible", 
    "horrible", "amazing", "incredible", "best", "worst", "stupid",
    "genius", "ridiculous", "outrageous", "disgusting", "pathetic",
    "brilliant", "gorgeous", "ugly", "evil", "heroic", "masterpiece",
    "tragic", "wonderful", "awful", "fantastic", "dreadful"
}

# Compile a fast regex pattern to detect any of these words as whole words
BIAS_PATTERN = re.compile(r'\b(?:' + '|'.join(SUBJECTIVE_WORDS) + r')\b', re.IGNORECASE)

def is_objective(text: str) -> bool:
    """
    Returns True if the text is deemed objective (lacks manipulative/subjective words).
    Returns False if it contains manipulative or heavily subjective language.
    """
    if BIAS_PATTERN.search(text):
        return False
    return True

def main():
    # 2. Initialize the tokenizer to track the exact token count
    # cl100k_base is the standard high-performance encoding for modern LLMs
    enc = tiktoken.get_encoding("cl100k_base")
    
    target_tokens = 6_000_000_000  # 6 Billion tokens
    current_tokens = 0
    output_file = "clean_non_manipulative_corpus.jsonl"
    
    print("Connecting to Hugging Face and starting dataset stream...")
    
    # 3. Stream the dataset using the modern, official Wikimedia namespace
    # streaming=True ensures we iterate through the data in memory without massive downloads
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    print(f"Extracting objective text to {output_file}. This will take some time...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if current_tokens >= target_tokens:
                print(f"\nTarget of {target_tokens:,} tokens reached. Stopping extraction.")
                break
                
            text = item.get("text", "")
            if not text:
                continue
                
            # Filter at the paragraph level rather than article level.
            # This prevents throwing away a massive factual article just because of one biased quote.
            paragraphs = text.split("\n")
            clean_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                # Skip very short fragments, UI artifacts, or empty lines
                if len(para) < 50: 
                    continue
                    
                # 4. Apply the neutrality filter
                if is_objective(para):
                    clean_paragraphs.append(para)
            
            # Reconstruct the cleaned, purely objective article
            if clean_paragraphs:
                clean_text = "\n".join(clean_paragraphs)
                
                # Count the tokens of the finalized clean text
                tokens = len(enc.encode(clean_text, disallowed_special=()))
                current_tokens += tokens
                
                # Save the clean data incrementally
                record = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "text": clean_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Print progress to the terminal every ~10 million tokens
                if current_tokens % 10_000_000 < tokens:
                    print(f"Collected {current_tokens:>,} / {target_tokens:>,} tokens...")

    print("Data collection pipeline complete!")

if __name__ == "__main__":
    main()