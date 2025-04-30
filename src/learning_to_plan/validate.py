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
    test_tasks = {t for t in tasks if t.type == task.Task.TaskType.TEST}
    assert len(test_tasks) > 0, f"No test tasks found in {data_file_path}."
    config.log(f"Found {len(test_tasks)} test tasks.", level=config.logging.INFO)
    tasks = tasks - test_tasks
    val_executable = "/utils/VAL/bin/Validate" # Adjust if necessary
    cmd_list = [
        val_executable,
        "-v",  # Verbose output
        "-t", "0.001", # Tolerance for numeric effects (if any)
        t._domain_file_path,
        t._instance_file_path,
        temp_plan_file
    ]
    
    for t in test_tasks:
        config.log(f"Validating generated plans for task {t._id}...", level=config.logging.DEBUG)
        for model in t._model_generated_plans.keys():
            for i, p in enumerate(t._model_generated_plans[model]):
                temp_plan_file = f".temp_plan_{i}.txt"
                is_plan_valid = False
                try:
                    # Write the plan to a temporary file
                    with open(temp_plan_file, "w") as f:
                        f.write(p)
                    config.log(f"Created temporary plan file: {temp_plan_file}", level=config.logging.DEBUG)
                    # Construct the validation command list for subprocess
                    
                    # Ensure the path to VAL executable is correct for your environment
                    config.log(f"Executing VAL command: {' '.join(cmd_list)}", level=config.logging.DEBUG)
                    # Execute the command using subprocess.run to capture output
                    result = subprocess.run(cmd_list, capture_output=True, text=True, check=False) 
                    config.log(f"VAL command executed with return code: {result.returncode}", level=config.logging.DEBUG)
                    for line in result.stdout.splitlines():
                        config.log(f"VAL output indicates success for plan {i} of task {t._id}.", level=config.logging.INFO)
                        if "Plan executed successfully" in line:
                            is_plan_valid = True
                            break
                    t.validate_plan(model, i, is_plan_valid)
                    if is_plan_valid:
                        config.log(f"Plan {i} of task {t._id} is valid.", level=config.logging.INFO)
                    else:
                        config.log(f"VAL output does not contain 'Plan executed successfully' for plan {i} of task {t._id}.", level=config.logging.WARNING)
                        # Log detailed output for debugging failed validations
                        config.log(f"VAL return code: {result.returncode}", level=config.logging.DEBUG)
                        config.log(f"VAL stdout:\n{result.stdout}", level=config.logging.DEBUG)
                        config.log(f"VAL stderr:\n{result.stderr}", level=config.logging.DEBUG)
                    tasks.add(t)
                    
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
        
    try:
        config.create_necessary_dirs(data_file_path)
        task.save_tasks_to_jsonl(tasks, data_file_path)
        config.log(f"Finished writing to {data_file_path}.", level=config.logging.INFO)
    except Exception as e:
        config.log(f"Error saving tasks to JSONL: {e}", level=config.logging.ERROR)
        raise e

    # -- Compute metrics --
    # Metrics
    # -1. for each task, if a plan is valid, add 1 to the number of valid plans. Divide this number by the number of tasks
    # -2. for each task, take the mean of the number of valid plans and the std of the number of valid plans. Take the mean of these two numbers across all tasks

    