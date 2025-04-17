from learning_to_plan.task import get_task_from_csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import learning_to_plan.config as config
import datetime

def build_finetuining_dataset(
    csv_path,
    train_output,
    validation_output,
    test_output,
    random_seed=42
):
    config.logging.info("Building finetuning dataset. Starting at %s", datetime.datetime.now())
    if os.path.exists(train_output) and os.path.exists(validation_output) and os.path.exists(test_output):
        e = f"Finetuning dataset files already exist: {train_output}, {validation_output}, {test_output}"
        config.logging.error(e)
        raise ValueError(e)

    if not os.path.exists(csv_path):
        e = f"CSV file not found: {csv_path}"
        config.logging.error(e)
        raise ValueError(e)

    df = pd.read_csv(csv_path)
    # Filter valid rows and separate into longer and basic plans
    valid_mask = (df["status"] == "ok") & (df["plan"].notna()) & (df["plan"].str.strip() != "")
    df_valid = df[valid_mask]

    if not len(df_valid) == 4400:
        e = f"CSV file must contain at least 4400 valid rows, but only {len(df_valid)} rows were found."
        config.logging.error(e)
        raise ValueError(e)

    # Extract longer plans and basic plans
    longer_df = df_valid[df_valid["is_longer_plan"] == True]
    basic_df = df_valid[df_valid["is_longer_plan"] == False]
    
    # Check if exactly 200 rows have is_longer_plan as True
    if len(longer_df) != 200:
        e = f"Expected 200 rows with 'is_longer_plan' as True, but found {len(longer_df)} rows."
        config.logging.error(e)
        raise ValueError(e)
    
    # Only split the basic dataset
    train_df, temp_df = train_test_split(
        basic_df, test_size=800, random_state=random_seed
    )
    validation_df, basic_test_df = train_test_split(
        temp_df, test_size=200, random_state=random_seed
    )
    test_df = pd.concat([longer_df, basic_test_df], ignore_index=True)

    def write_dataset(df, output_path):
        config.create_necessary_dirs(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                task = get_task_from_csv(row)
                prompt = task.build_prompt()
                f.write(prompt + "\n")
    
    write_dataset(train_df, train_output)
    write_dataset(validation_df, validation_output)
    write_dataset(test_df, test_output)
    config.logging.info("Finished building finetuning dataset. Ending at %s", datetime.datetime.now())