from learning_to_plan.task import get_task_from_csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from learning_to_plan.config import create_necessary_dirs


def build_finetuining_dataset(
    csv_path,
    train_output,
    test_output,
    test_size=0.2,
    random_seed=42
):
    if os.path.exists(train_output) and os.path.exists(test_output):
        print(f"Finetuning dataset files already exist: {train_output}, {test_output}")
        return

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_valid = df[(df["status"] == "ok") & (df["plan"].notna()) & (df["plan"].str.strip() != "")]

    if len(df_valid) == 0:
        print("No valid rows found in dataset.")
        return
    

    train_df, test_df = train_test_split(
        df_valid, test_size=test_size, random_state=random_seed
    )

    def write_dataset(df, output_path):
        create_necessary_dirs(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                task = get_task_from_csv(row)
                prompt = task.build_prompt()
                f.write(prompt + "\n")
    
    write_dataset(train_df, train_output)
    write_dataset(test_df, test_output)