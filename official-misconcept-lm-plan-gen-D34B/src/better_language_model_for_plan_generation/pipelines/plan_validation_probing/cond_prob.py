import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import dotenv
import os 

TOKEN_ID_FOR_QWEN = {
    '\n': 198,
    ' ': 220,
    ' \n': 715,
    ' WRONG': 98660,
}

def compare_phrase_probabilities(model, tokenizer, context):
    # in our case, 
    # for CoT, if the model predicts " WRONG", it identifies as a wrong action, if it predicts "\n", it continues to the next action
    # for no CoT, if the model predicts " WRONG", it identifies as a wrong action, if it predicts "\n", it continues to the next action
    
    # ! first phrase is to identify the wrong action, second phrase is to identify the newline character ( no mistake)
    def calculate_last_phrase_metrics(input_ids):
        # input_ids = torch.tensor([input_ids]) # batch size 1, somehow encode already gives a batch size of 1 so we comment this line
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        
        # assert last token is the last token of the phrase
        
        next_token_logits = logits[0, -1] # should be the previous token
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        mistake_token_prob = next_token_probs[TOKEN_ID_FOR_QWEN[' WRONG']].item()
        no_mistake_token_prob_1 = next_token_probs[TOKEN_ID_FOR_QWEN[' \n']].item()
        no_mistake_token_prob_2 = next_token_probs[TOKEN_ID_FOR_QWEN['\n']].item()
        no_mistake_token_prob_3 = next_token_probs[TOKEN_ID_FOR_QWEN[' ']].item()
        # get max 
        no_mistake_token_prob = max(no_mistake_token_prob_1, no_mistake_token_prob_2, no_mistake_token_prob_3)
        
        return mistake_token_prob, no_mistake_token_prob

    tokens = tokenizer.encode(context.rstrip(), add_special_tokens=False, return_tensors="pt").to('cuda')
    
    mistake_token_prob, no_mistake_token_prob = calculate_last_phrase_metrics(tokens)
    
    return mistake_token_prob, no_mistake_token_prob

if __name__ == "__main__":
    dotenv.load_dotenv(os.path.join(os.environ['WORKING_DIR'], ".env"))
    print(os.environ['HF_HOME'])
    # Load pre-trained model and tokenizer
    model_name = "Qwen/Qwen2-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Compare the phrases with the given context
    context = "This is"
    mistake_token_prob, no_mistake_token_prob = compare_phrase_probabilities(model, tokenizer, context)

    print(f"Metrics for identifying mistake given '{context}':")
    print(f"  Probability per token: {mistake_token_prob:.9f}")

    print(f"\nMetrics for NOT identifying mistake given '{context}':")
    print(f"  Probability per token: {no_mistake_token_prob:.9f}")

    # Compare the phrases
    print("\nComparison:")

    print(f"Probability ratio over MISTAKE/NO MISTAKE: {mistake_token_prob / no_mistake_token_prob:.4f}")
    # their ratio values are actually exactly the same. 

    # Metrics for identifying mistake given 'This is':
    #   Probability per token: 0.000000264

    # Metrics for NOT identifying mistake given 'This is':
    #   Probability per token: 0.001388321

    # Comparison:
    # Probability ratio over MISTAKE/NO MISTAKE: 0.0002