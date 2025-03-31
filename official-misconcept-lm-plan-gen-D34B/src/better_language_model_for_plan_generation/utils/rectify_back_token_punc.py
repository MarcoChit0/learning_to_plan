from tqdm.auto import tqdm
from glob import glob 
import os 

# run this script after generating the dataset
if __name__ == "__main__":
    dataset_dir = os.path.join(os.environ['WORKING_DIR'], "data/03_primary/llm_plan_generation_dataset")
    dataset_dir_qwen = os.path.join(os.environ['WORKING_DIR'], "data/03_primary/llm_plan_generation_dataset_qwen")


    jsonl_files = glob(os.path.join(dataset_dir, "**/*.jsonl"), recursive=True)

    jsonl_files_qwen = glob(os.path.join(dataset_dir_qwen, "**/*.jsonl"), recursive=True)

    print(f"Number of jsonl files in the dataset: {len(jsonl_files)}")
    print(f"Number of jsonl files in the dataset_qwen: {len(jsonl_files_qwen)}")

    # for jsonl_file in tqdm(jsonl_files, desc="Replace back token punctuations on raw dataset"):
    #     with open(jsonl_file, "r") as infile, open(jsonl_file+".bak", "w") as outfile:
    #         for line in infile:
    #             modified_line = line.replace("  WRONG", " WRONG")
    #             outfile.write(modified_line)
                
    #     # replace the original file
    #     os.rename(jsonl_file+".bak", jsonl_file)
        
    # print("Done! Part 1/2")
        
    # for jsonl_file in tqdm(jsonl_files_qwen, desc="Replace back token punctuations on qwen dataset"):
    #     with open(jsonl_file, "r") as infile, open(jsonl_file+".bak", "w") as outfile:
    #         for line in infile:
    #             modified_line = line.replace("  WRONG", " WRONG")
    #             outfile.write(modified_line)
                
    #     # replace the original file
    #     os.rename(jsonl_file+".bak", jsonl_file)
        
    # print("Done! Part 2/2")

    for jsonl_file in tqdm(jsonl_files_qwen, desc="solve t4 count issue"):
        if 'accu_t2_t4' in jsonl_file:
            with open(jsonl_file, "r") as infile, open(jsonl_file+".bak", "w") as outfile:
                for line in infile:
                    modified_line = line.replace("  <COUNT>", " <COUNT>").replace("</COUNT>  ", "</COUNT> ") # remove 1 space for 2 spaces
                    outfile.write(modified_line)
            # replace the original file
            os.rename(jsonl_file+".bak", jsonl_file)
        elif 'accu_t4' in jsonl_file:
            with open(jsonl_file, "r") as infile, open(jsonl_file+".bak", "w") as outfile:
                for line in infile:
                    modified_line = line.replace("   <COUNT>", "  <COUNT>").replace("</COUNT>   ", "</COUNT>  ") # remove 1 space for 3 spaces
                    outfile.write(modified_line)
            # replace the original file
            os.rename(jsonl_file+".bak", jsonl_file)