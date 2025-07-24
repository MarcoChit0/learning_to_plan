import subprocess
import dotenv
import os
import sys 

# TODO: CHECK WHETHER THE LAST RUN RESULTED IN ERROR AND IF SO WHETHER IT WAS DUE TO RATE LIMITING. IF IT WAS DUE TO RATE LIMITING, THEN REMOVE THE GENERATED PROBLEMS WITH ERROR, CHANGE THE API KEY AND RE-RUN THE GENERATION SCRIPT.
def main():
    """
    Main function to run the generation script.
    """
    dotenv.load_dotenv()

    number_of_problems = 20
    number_of_samples = 1
    prompt_type = "landmarks"
    task_type = "outofdistribution"
    thinking_style = "none"

    RATE_LIMIT_GEMINI = 250
    RATE_LIMITE_GEMMA = 14000

    google_api_keys = []
    i = 0
    while True:
        try:
            google_api_key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if not google_api_key:
                break
            google_api_keys.append(google_api_key)
            i += 1
        except Exception as e:
            print(f"Error loading API key GOOGLE_API_KEY_{i}: {e}")
            break
    
    CONFIGS = [
        "src/configs/generate/gemini-thinking.json"
    ]

    DOMAINS = [
        "barman", "blocksworld", "childsnack", "depots", 
        "driverlog", "grippers", "logistics", "satellite"
    ]

    for config in CONFIGS:
        print(f"Using configuration: {config}")
        print("----------------------------------------")
        
        for domain in DOMAINS:
            print("----------------------------------------")
            print(f"Generating problems for domain: {domain}")
            
            command = [
                sys.executable,  # Use the same python interpreter that is running this script
                "src/learning_to_plan/main.py",
                "--generate",
                "-c", config,
                "-n", str(number_of_problems),
                "-d", domain,
                "--task_type", task_type,
                "--prompt_type", prompt_type,
                "-s", str(number_of_samples),
                "--thinking_style", thinking_style,
                "--google_api_key", google_api_keys[1]
            ]
            
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command for domain {domain} with config {config}: {e}")
            except FileNotFoundError:
                print(f"Error: 'src/learning_to_plan/main.py' not found. Make sure you are running this script from the correct directory.")
                sys.exit(1)

            print(f"Finished generating problems for domain: {domain}")
            print("----------------------------------------")
            
        print(f"Finished generating problems with configuration: {config}")
        print("----------------------------------------")

if __name__ == "__main__":
    main()