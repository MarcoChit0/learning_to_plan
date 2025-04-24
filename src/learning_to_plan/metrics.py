import os
from learning_to_plan import config
from learning_to_plan import task
import datetime
import subprocess
# domin_path = "data/raw/blocksworld/generated_domain.pddl"
# problem_path = "data/raw/blocksworld/generated_basic/instance-0.pddl"
# plan ="""(unstack a c)
# (put-down a)
# (pick-up b)
# (stack b c)
# (pick-up a)
# (stack a b)"""

# with open("plan_blocksworld_instance-0.txt", "w") as f:
#     f.write(plan)


# val_command = f"../VAL/bin/Validate -v -t 0.001 {domin_path} {problem_path} plan_blocksworld_instance-0.txt"
# os.system(val_command)

def validate_plans(data_file_path):
    config.log(f"Starting plan validation at {datetime.datetime.now()}.", level=config.logging.INFO)

    assert os.path.exists(data_file_path), f"Data file {data_file_path} does not exist."
    tasks:set[task.Task] = task.load_tasks(data_file_path)
    assert len(tasks) > 0, f"Data file {data_file_path} is empty."
    config.log(f"Loaded {len(tasks)} tasks from {data_file_path}.", level=config.logging.INFO)

    validation_results = {} # Optional: Store results (e.g., {task_id: [bool, bool, ...]})
    for t in tasks:
        task_validation_results = []
        config.log(f"Validating plans for task {t._id}...", level=config.logging.DEBUG)
        for i, p in enumerate(t._generated_plans):
            temp_plan_file = f".temp_plan_{i}.txt"
            plan_valid = False
            try:
                # Write the plan to a temporary file
                with open(temp_plan_file, "w") as f:
                    f.write(p)
                config.log(f"Created temporary plan file: {temp_plan_file}", level=config.logging.DEBUG)
                # Construct the validation command list for subprocess
                # Ensure the path to VAL executable is correct for your environment
                val_executable = "/utils/VAL/bin/Validate" # Adjust if necessary
                cmd_list = [
                    val_executable,
                    "-v",  # Verbose output
                    "-t", "0.001", # Tolerance for numeric effects (if any)
                    t._domain_file_path,
                    t._instance_file_path,
                    temp_plan_file
                ]
                config.log(f"Executing VAL command: {' '.join(cmd_list)}", level=config.logging.DEBUG)
                # Execute the command using subprocess.run to capture output
                result = subprocess.run(cmd_list, capture_output=True, text=True, check=False) # check=False prevents raising CalledProcessError
                # Check the standard output for the "Successful plans:" or "Plan valid" string (common VAL success indicators)
                # The user asked for "Success", let's check for that specifically, but be aware VAL might output something else.
                # Consider checking result.returncode == 0 as well for a more robust check.
                if "Success" in result.stdout: # Checking specifically for "Success" as requested
                    config.log(f"VAL output indicates success for plan {i} of task {t._id}.", level=config.logging.INFO)
                    plan_valid = True
                # You might want a more specific check based on actual VAL output, e.g.:
                # if "Plan valid" in result.stdout or "Successful plans:" in result.stdout:
                #    config.log(f"VAL indicated plan {i} of task {t._id} is valid.", level=config.logging.INFO)
                #    plan_valid = True
                else:
                    config.log(f"VAL output does not contain 'Success' for plan {i} of task {t._id}.", level=config.logging.WARNING)
                    # Log detailed output for debugging failed validations
                    config.log(f"VAL return code: {result.returncode}", level=config.logging.DEBUG)
                    config.log(f"VAL stdout:\n{result.stdout}", level=config.logging.DEBUG)
                    config.log(f"VAL stderr:\n{result.stderr}", level=config.logging.DEBUG)
            except FileNotFoundError:
                 config.log(f"Error: VAL executable not found at '{val_executable}'. Cannot validate plan {i} for task {t._id}.", level=config.logging.ERROR)
            except Exception as e:
                 config.log(f"An unexpected error occurred during validation for plan {i} of task {t._id}: {e}", level=config.logging.ERROR)
            finally:
                if os.path.exists(temp_plan_file):
                    try:
                        os.remove(temp_plan_file)
                        config.log(f"Removed temporary plan file: {temp_plan_file}", level=config.logging.DEBUG)
                    except OSError as e:
                        config.log(f"Error removing temporary file {temp_plan_file}: {e}", level=config.logging.ERROR)
            task_validation_results.append(plan_valid)
        validation_results[t._id] = task_validation_results
        config.log(f"Finished validation for task {t._id}. Results: {task_validation_results}", level=config.logging.INFO)