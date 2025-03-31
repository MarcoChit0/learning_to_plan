import os
import sys
from glob import glob
from pathlib import Path

cur_dir = "/home/sukaih/Extrastorage/PhDProject_Ext/BetterLMforPlanGen/offical-misconcept-lm-plan-gen/data/03_primary/llm_plan_generation_dataset_qwen"

target_dir = "/home/sukaih/Extrastorage/PhDProject_Ext/BetterLMforPlanGen/better-language-model-for-plan-generation/data/03_primary/llm_plan_generation_dataset_qwen"

want_files = glob(target_dir + "/**/chunk_0.jsonl", recursive=True)

print(len(want_files))
for file in want_files:
    # replace the target_dir with the current directory
    new_file = file.replace(target_dir, cur_dir)
    Path(new_file).parent.mkdir(parents=True, exist_ok=True)
    # cp the file
    if "test" in new_file:
        os.system("cp " + file + " " + new_file)