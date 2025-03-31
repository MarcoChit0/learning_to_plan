from glob import glob 
import os 
from pathlib import Path 
from tqdm.auto import tqdm
import re

instance_files = glob(os.path.join(os.path.dirname(__file__), "**/instance-*.pddl"))
print(len(instance_files))

for fp in tqdm(instance_files):
    with open(fp, 'r') as f:
        data = f.read()
    
    # replace all the "object" to "obj"
    
    pattern = re.compile(r"\bobject\b")
    data = re.sub(pattern, "obj", data)
    
    with open(fp, 'w') as f:
        f.write(data)
        
    