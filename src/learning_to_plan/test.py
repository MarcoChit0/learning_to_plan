import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import learning_to_plan.config as config
from dotenv import load_dotenv
import datetime
import json
import pandas as pd
import numpy as np

def run_evaluation_procedure(model_dir, test_file):
    """
    Evaluate a language model using the test data.
    
    For each instance in test_file (a JSONL file), the function:
      - Separates the prompt from the ground-truth plan (using the "## Plan." marker).
      - Sends the prompt (without the ground-truth plan) to the model to generate a plan.
      - Compares the generated plan with the ground truth exactly.
      
    After evaluation, the result is saved (or appended) into a CSV file in model_dir with:
      - timestamp,
      - accuracy (in percentage),
      - the total number of training epochs (from the config),
      - and the model parameters (as a canonical JSON string).
      
    If the model's parameters (saved in training_params.json) differ from what is stored
    in the CSV file, the CSV file is overwritten, otherwise the new result is appended.
    """
    config.logging.info("Starting evaluation using model from %s at %s", model_dir, datetime.datetime.now())
    load_dotenv()
    autentication_token = os.getenv("HUGGINGFACE_TOKEN")
    
    # Load tokenizer and model from the checkpoint directory.
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_auth_token=autentication_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
        use_auth_token=autentication_token
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the test dataset.
    dataset = load_dataset("json", data_files={"test": test_file})
    if len(dataset["test"]) == 0:
        config.logging.error("Test dataset is empty.")
        raise ValueError("Test dataset is empty.")
    
    total_instances = len(dataset["test"])
    data:list[dict[str,object]] = []
    
    for instance in dataset["test"]:
        full_prompt = instance["prompt"]
        # Split the instance at the "## Plan." marker.
        if "## Plan." not in full_prompt:
            config.logging.warning("Test instance missing '## Plan.' marker. Skipping instance.")
            continue

        prompt_part, _, plan_part = full_prompt.partition("## Plan.")
        input_prompt = prompt_part.strip() + "\n## Plan.\n\n"
        ground_truth = plan_part.strip()
        
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MODEL_TRAINING_CONFIG["max_new_tokens"],
            do_sample=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Count the number of correct actions in the generated plan. Longest common sequence, where each element in the sequence is an action.
        
        generated_plan = generated_text.split("\n")
        generated_plan = [action.strip() for action in generated_plan if action.strip()]
        ground_truth_plan = ground_truth.split("\n")
        ground_truth_plan = [action.strip() for action in ground_truth_plan if action.strip()]

        C = np.zeros((len(generated_plan) + 1, len(ground_truth_plan) + 1))
        for i in range(1, len(generated_plan) + 1):
            for j in range(1, len(ground_truth_plan) + 1):
                if generated_plan[i - 1] == ground_truth_plan[j - 1]:
                    C[i][j] = C[i - 1][j - 1] + 1
                else:
                    C[i][j] = max(C[i - 1][j], C[i][j - 1])
        
        longest_common_sequence = C[len(generated_plan)][len(ground_truth_plan)]
        data.append({
            "instance": instance,
            "correct" : 1 if longest_common_sequence == len(ground_truth_plan) else 0,
            "lcs" : longest_common_sequence,
            "lcs_ratio" : longest_common_sequence / len(ground_truth_plan),
            "generated_plan_length": len(generated_plan),
            "ground_truth_plan_length": len(ground_truth_plan)
        })
        config.logging.info("Evaluated instance %d/%d", len(data), total_instances)
        config.logging.info("\tGenerated plan lenght: %s", len(generated_text))
        config.logging.info("\tGround truth plan lenght: %s", len(ground_truth))
        config.logging.info("\tLongest common sequence: %s", longest_common_sequence)
        config.logging.info("\tLongest common sequence ratio: %s", longest_common_sequence / len(ground_truth_plan))
        
    metrics_file_path = os.path.join(model_dir, config.TEST_METRICS_FILE_NAME)
    if os.path.exists(metrics_file_path):
        df = pd.read_csv(metrics_file_path)
        index = df.index[-1]
    else:
        df = pd.DataFrame(columns=["timestamp", "trained_epochs", "accuracy", "mean_lcs", "std_lcs", "mean_lcs_ratio", "std_lcs_ratio", "mean_generated_plan_length", "std_generated_plan_length"])
        index = 0

    data_file_path = os.path.join(model_dir, config.TEST_DATA_FILE_NAME.format(index=index+1))
    data_df = pd.DataFrame(data)
    data_df.to_csv(data_file_path, index=False)

    # Compute metrics
    accuracy = sum([d["correct"] for d in data]) / total_instances
    accuracy = accuracy * 100
    mean_lcs = np.mean([d["lcs"] for d in data])
    std_lcs = np.std([d["lcs"] for d in data])
    mean_lcs_ratio = np.mean([d["lcs_ratio"] for d in data])
    std_lcs_ratio = np.std([d["lcs_ratio"] for d in data])
    mean_generated_plan_length = np.mean([d["generated_plan_length"] for d in data])
    std_generated_plan_length = np.std([d["generated_plan_length"] for d in data])
    mean_ground_truth_plan_length = np.mean([d["ground_truth_plan_length"] for d in data])
    std_ground_truth_plan_length = np.std([d["ground_truth_plan_length"] for d in data])

    new_row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_epochs": config.MODEL_TRAINING_CONFIG["num_train_epochs"],
        "accuracy": accuracy,
        "mean_lcs": mean_lcs,
        "std_lcs": std_lcs,
        "mean_lcs_ratio": mean_lcs_ratio,
        "std_lcs_ratio": std_lcs_ratio,
        "mean_generated_plan_length": mean_generated_plan_length,
        "std_generated_plan_length": std_generated_plan_length,
    }
    df = df.append(new_row, ignore_index=True)
    df.to_csv(metrics_file_path, index=False)
    config.logging.info("Saved evaluation results to %s", metrics_file_path)
