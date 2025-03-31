from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.tools import EncodingVisualizer
from IPython.display import display, HTML
import html
import random 

def generate_pastel_color():
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(180, 255),
        random.randint(180, 255),
        random.randint(180, 255)
    )

def visualize_tokens(text, tokenizer):
    # Tokenize the text
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']

    # Create HTML for visualization
    html_parts = []
    last_end = 0
    colors = {}

    for (start, end), token in zip(offset_mapping, tokens):
        if start > last_end:
            html_parts.append(html.escape(text[last_end:start]))
        
        if token not in colors:
            colors[token] = generate_pastel_color()
        
        color = colors[token]
        html_parts.append(f'<span style="background-color: {color}; border: 1px solid black;" title="{html.escape(token)}">{html.escape(text[start:end])}</span>')
        last_end = end

    if last_end < len(text):
        html_parts.append(html.escape(text[last_end:]))

    # Combine HTML parts
    html_output = ''.join(html_parts)
    
    # Display in Jupyter notebook
    display(HTML(f'<p style="font-family: monospace;">{html_output}</p>'))
    
    # Print tokens
    print("Tokens:", tokens)


if __name__ == "__main__":
    # use IPython 
    
    text = "<GOAL> My goal is to have that shot_136 contains cocktail_321, shot_1 contains cocktail_321. <COUNT> 4 </COUNT>  <PRECON> We can pour the cocktail_321 of a shaker_123 into a shot_136 only if the shot_136 is empty and clean. We can pour the cocktail_321 of a shaker_123 into a shot_136 only if we are holding the shaker_123 and the shaker_123 contains a cocktail_321. </PRECON> pour from shaker_123 to shot_136 containing cocktail_321 using left_hand from level_1 to level_0 <EFFECT> Once we pour the cocktail_321 of a shaker_123 into a shot_136, the shot_136 is not clean. Once we pour the cocktail_321 of a shaker_123 into a shot_136, the shot_136 contains the cocktail_321 and the shot_136 is not empty. Once we pour the cocktail_321 of a shaker_123 into a shot_136, the shaker_123's fill level level_1 will decrement one level level_0. </EFFECT> WRONG \n <GOAL> My goal is to have that shot_136 contains cocktail_321, shot_1 contains cocktail_321. <COUNT> 4 </COUNT>  <PRECON> We can pour the ingredient_315 of a shot_290 into a used shaker_123 only if the shaker_123 has not been shaken yet. We can pour the ingredient_315 of a shot_290 into a used shaker_123 only if we are holding the shot_290 and the shot_290 contains some ingredient_315. </PRECON> pour from shot_290 containing ingredient_315 to used shaker_123 using right_hand from level_1 to level_2 <EFFECT> Once we pour the ingredient_315 of a shot_290 into a used shaker_123, the shot_290 does not contain any ingredient_315 and is empty. Once we pour the ingredient_315 of a shot_290 into a used shaker_123, the shaker_123 contains the ingredient_315. Once we pour the ingredient_315 of a shot_290 into a used shaker_123, the shaker_123's fill level level_1 will increment one level level_2. </EFFECT> \n"
    
    
    
    # Usage
    test_text = "My plan is as follows:\n<PLAN>\ngrasp the shaker_152 using left_hand\ngrasp the shot_460 using right_hand\nleave the shaker_152 using left_hand\nfill the shot_460 on right_hand with ingredient_191 using dispenser_255 when left_hand is empty\ngrasp the shaker_152 using left_hand\nleave the shaker_152 using left_hand WRONG \npour from shot_460 containing ingredient_191 to clean shaker_152 using right_hand from level_0 to level_1\npour from shaker_152 to shot_460 containing cocktail_84 using left_hand from level_1 to level_0 WRONG \nleave the shaker_152 using left_hand\nclean the shot_460 on right_hand used for ingredient_191 when left_hand is empty\nleave the shot_460 using right_hand WRONG \nfill the shot_460 on right_hand with ingredient_453 using dispenser_149 when left_hand is empty\npour from shot_460 containing ingredient_453 to used shaker_152 using right_hand from level_1 to level_2\npour from shaker_152 to shot_249 containing cocktail_84 using left_hand from level_2 to level_1 WRONG \nclean the shot_460 on right_hand used for ingredient_453 when left_hand is empty\npour from shaker_152 to shot_249 containing cocktail_84 using left_hand from level_2 to level_1 WRONG \ngrasp the shaker_152 using left_hand\nleave the shot_460 using right_hand\nshake shaker_152 on left_hand containing ingredient_453 and ingredient_191 to get cocktail_84 when right_hand is empty\npour from shaker_152 to shot_249 containing cocktail_84 using left_hand from level_2 to level_1\npour from shaker_152 to shot_460 containing cocktail_84 using left_hand from level_1 to level_0\n</PLAN>"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    visualize_tokens(test_text, tokenizer)
    
    