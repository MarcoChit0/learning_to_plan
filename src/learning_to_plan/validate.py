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
    tasks:set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
    assert len(tasks) > 0, f"Data file {data_file_path} is empty."
    config.log(f"Loaded {len(tasks)} tasks from {data_file_path}.", level=config.logging.INFO)
    test_tasks = {t for t in tasks if t._type == task.Task.TaskType.TEST}
    assert len(test_tasks) > 0, f"No test tasks found in {data_file_path}."
    config.log(f"Found {len(test_tasks)} test tasks.", level=config.logging.INFO)
    tasks = tasks - test_tasks
    val_executable = "utils/VAL/build/bin/Validate" # Adjust if necessary

    for t in test_tasks:
        config.log(f"Validating generated plans for task {t._id}...", level=config.logging.DEBUG)
        for model in t._model_generated_plans.keys():
            for i, nlp_class in enumerate(t._model_generated_plans[model]):
                temp_plan_file = f".temp_plan_{t._id}_{model}_{i}.txt".replace(" ", "_").replace("/", "_")
                is_plan_valid = False
                try:
                    with open(temp_plan_file, "w") as f:
                        f.write(nlp_class._plan)
                    config.log(f"Created temporary plan file: {temp_plan_file}", level=config.logging.DEBUG)
                    cmd_list = [
                        val_executable,
                        "-v",  # Verbose output
                        "-t", "0.001", # Tolerance for numeric effects (if any)
                        t._domain_file_path,
                        t._instance_file_path,
                        temp_plan_file
                    ]
                    config.log(f"Executing VAL command: {' '.join(cmd_list)}", level=config.logging.DEBUG)
                    result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
                    os.remove(temp_plan_file)
                    config.log(f"VAL command executed with return code: {result.returncode}", level=config.logging.DEBUG)
                    for line in result.stdout.splitlines():

                        if "Plan valid" in line:
                            is_plan_valid = True
                            break
                    t.validate_plan(model, i, is_plan_valid)
                    if is_plan_valid:
                        config.log(f"Plan {i} of task {t._id} is valid.", level=config.logging.INFO)
                    else:
                        config.log(f"VAL output does not contain 'Plan valid' for plan {i} of task {t._id}.", level=config.logging.WARNING)
                        config.log(f"VAL return code: {result.returncode}", level=config.logging.DEBUG)
                        config.log(f"VAL stdout:\n{result.stdout}", level=config.logging.DEBUG)
                        config.log(f"VAL stderr:\n{result.stderr}", level=config.logging.DEBUG)
                    tasks.add(t)
                    
                except Exception as e:
                    config.log(f"Error validating plan {i} of task {t._id}: {e}", level=config.logging.ERROR)
                    raise e

    try:
        task.save_tasks_to_jsonl(tasks, data_file_path)
        config.log(f"Finished writing to {data_file_path}.", level=config.logging.INFO)
    except Exception as e:
        config.log(f"Error saving tasks to JSONL: {e}", level=config.logging.ERROR)
        raise e