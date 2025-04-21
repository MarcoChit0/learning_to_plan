import os

domin_path = "data/raw/blocksworld/generated_domain.pddl"
problem_path = "data/raw/blocksworld/generated_basic/instance-0.pddl"
plan ="""(unstack a c)
(put-down a)
(pick-up b)
(stack b c)
(pick-up a)
(stack a b)"""

with open("plan_blocksworld_instance-0.txt", "w") as f:
    f.write(plan)


val_command = f"../VAL/bin/Validate -v -t 0.001 {domin_path} {problem_path} plan_blocksworld_instance-0.txt"
os.system(val_command)
