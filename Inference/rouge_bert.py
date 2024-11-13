from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoTokenizer # Import AutoTokenizer
import torch
# Sample lists of Bengali text (references and candidates)
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')  # Required for METEOR
import pandas as pd
test = pd.read_csv('/content/test.csv')

alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Load the model and tokenizer
model_name = "/content/drive/MyDrive/CUET/checkpoint-20000"
# Initialize the tokenizer here 
tokenizer = AutoTokenizer.from_pretrained(model_name) # Assuming model_name is defined
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(example):
    # ... (rest of your code)
    # Retrieve question and answer from the example
    instruction = "Please provide a summary of the following article"
    question = example["article"]
    answer = example["summary"]

    # Check the structure and content of the example
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")

    # Construct the formatted prompt text
    prompt_text = alpaca_prompt.format(instruction, question, answer) + EOS_TOKEN

    # Return the formatted prompt text as a dictionary
    return {"text": prompt_text}


max_seq_length = 1024  # Set this as per your model configuration
dtype = torch.float16  # Adjust based on your setup
load_in_4bit = True  # Set as needed for your setup

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

# Prepare a list to store all generated responses
responses = []

# Assuming `test` is your dataset with an 'article' column (for example)
# Use `test.iterrows()` to iterate through the rows of the DataFrame
for i, row in test.iterrows(): 
    # Now `row` is a Pandas Series representing a row of data
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Please provide a summary of the following article",  # Instruction
                row["article"],  # The article content to summarize
                ""  # Leave output blank for generation
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    # Generate text and keep the output in a variable
    generated_ids = model.generate(**inputs, max_new_tokens=2024)

    # Decode the generated text and store it in a variable
    generated_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Extract the response after the separator and append it to the list
    try:
        response = generated_output.split("### Response:")[1]
    except IndexError:
        response = "No response generated"

    responses.append(response)

ground_truth = []
for i, row in test.iterrows(): 
    ground_truth.append(row["summary"])
# Remove newline characters
model_summary = [text.replace('\n', '') for text in responses]

def tokenize(text):
    # Tokenize the text into words (for ROUGE-1)
    return text.split()

def ngrams(tokens, n):
    # Create n-grams (for ROUGE-2 or higher n-grams)
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def precision_recall_f1(reference_ngrams, candidate_ngrams):
    # Precision = Overlap in candidate ngrams / Total candidate ngrams
    # Recall = Overlap in candidate ngrams / Total reference ngrams
    overlap = len(set(reference_ngrams) & set(candidate_ngrams))
    precision = overlap / len(candidate_ngrams) if candidate_ngrams else 0
    recall = overlap / len(reference_ngrams) if reference_ngrams else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def rouge_l(reference_tokens, candidate_tokens):
    # Find Longest Common Subsequence (LCS)
    m, n = len(reference_tokens), len(candidate_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Building the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference_tokens[i - 1] == candidate_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Length of the LCS
    lcs_length = dp[m][n]
    
    # ROUGE-L precision, recall, and F1 score
    precision = lcs_length / len(candidate_tokens) if candidate_tokens else 0
    recall = lcs_length / len(reference_tokens) if reference_tokens else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

# Sample lists of Bengali text (references and candidates)
references = model_summary

candidates = ground_truth

# Initialize variables to accumulate scores
total_precision_rouge1 = total_recall_rouge1 = total_f1_rouge1 = 0
total_precision_rouge2 = total_recall_rouge2 = total_f1_rouge2 = 0
total_precision_rouge_l = total_recall_rouge_l = total_f1_rouge_l = 0
num_pairs = len(references)

# Loop over the references and candidates
for ref, cand in zip(references, candidates):
    # Tokenize the reference and candidate
    reference_tokens = tokenize(ref)
    candidate_tokens = tokenize(cand)

    # ROUGE-1 (Unigram)
    reference_rouge1 = reference_tokens
    candidate_rouge1 = candidate_tokens
    precision_rouge1, recall_rouge1, f1_rouge1 = precision_recall_f1(reference_rouge1, candidate_rouge1)

    # ROUGE-2 (Bigram)
    reference_rouge2 = ngrams(reference_tokens, 2)
    candidate_rouge2 = ngrams(candidate_tokens, 2)
    precision_rouge2, recall_rouge2, f1_rouge2 = precision_recall_f1(reference_rouge2, candidate_rouge2)

    # ROUGE-L (Longest Common Subsequence)
    precision_rouge_l, recall_rouge_l, f1_rouge_l = rouge_l(reference_tokens, candidate_tokens)

    # Accumulate the scores
    total_precision_rouge1 += precision_rouge1
    total_recall_rouge1 += recall_rouge1
    total_f1_rouge1 += f1_rouge1
    
    total_precision_rouge2 += precision_rouge2
    total_recall_rouge2 += recall_rouge2
    total_f1_rouge2 += f1_rouge2
    
    total_precision_rouge_l += precision_rouge_l
    total_recall_rouge_l += recall_rouge_l
    total_f1_rouge_l += f1_rouge_l

# Calculate the average scores
avg_precision_rouge1 = total_precision_rouge1 / num_pairs
avg_recall_rouge1 = total_recall_rouge1 / num_pairs
avg_f1_rouge1 = total_f1_rouge1 / num_pairs

avg_precision_rouge2 = total_precision_rouge2 / num_pairs
avg_recall_rouge2 = total_recall_rouge2 / num_pairs
avg_f1_rouge2 = total_f1_rouge2 / num_pairs

avg_precision_rouge_l = total_precision_rouge_l / num_pairs
avg_recall_rouge_l = total_recall_rouge_l / num_pairs
avg_f1_rouge_l = total_f1_rouge_l / num_pairs



references = model_summary
predictions = ground_truth

# BLEU Score
bleu_scores = [sentence_bleu([ref], pred) for ref, pred in zip(references, predictions)]
avg_bleu_score = sum(bleu_scores) / len(bleu_scores)


# Tokenize Bengali Text for METEOR
tokenized_references = [[ref.split()] for ref in references]  # Tokenize each reference sentence
tokenized_predictions = [pred.split() for pred in predictions]

# METEOR Score
meteor_scores = [meteor_score(ref, pred) for ref, pred in zip(tokenized_references, tokenized_predictions)]
avg_meteor_score = sum(meteor_scores) / len(meteor_scores)

# BERTScore (Change 'en' to 'bn' for Bengali if supported)
P, R, F1 = bert_score(predictions, references, lang="bn")
avg_bert_score = F1.mean().item()

# Print results
print(f"Average BLEU Score :  {avg_bleu_score:.2f}")
print(f"Average METEOR Score :  {avg_meteor_score:.2f}")
print(f"Average BERTScore F1 :  {avg_bert_score:.2f}")

# Print the overall results with 2 decimal points
print(f"Overall ROUGE-1 (Unigram): Precision={avg_precision_rouge1:.2f}, Recall={avg_recall_rouge1:.2f}, F1={avg_f1_rouge1:.2f}")
print(f"Overall ROUGE-2 (Bigram): Precision={avg_precision_rouge2:.2f}, Recall={avg_recall_rouge2:.2f}, F1={avg_f1_rouge2:.2f}")
print(f"Overall ROUGE-L (Longest Common Subsequence): Precision={avg_precision_rouge_l:.2f}, Recall={avg_recall_rouge_l:.2f}, F1={avg_f1_rouge_l:.2f}")
