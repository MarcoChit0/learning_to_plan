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
    
    
    data = data.replace('GroundStation', 'GroundStation'.lower())
    data = data.replace('Star', 'Star'.lower())
    data = data.replace('Planet', 'Planet'.lower())
    data = data.replace('Phenomenon', 'Phenomenon'.lower())
    
    
    
    with open(fp, 'w') as f:
        f.write(data)
        
    