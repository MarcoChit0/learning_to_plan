from pathlib import Path
import os 
import jsonlines
from copy import deepcopy
from datasets import load_dataset
from icecream import ic
import enum
from enum import IntEnum
from enum import auto


class CONFIG_TYPES(enum.Enum):
    # "type_id": ['t0', 'accu-t1', 'accu-t2', 'accu-t3', 'accu-t4', 'accu-t4', 'accu-t1+t4', 'accu-t2+t4'] # t0 is the raw prompt, t1 is the prompt with shuffled information (data augmentation), t2 is the lite CoT in [PLAN] section, t3 is the dense CoT information in [PLAN] section, t4 contains the "scratchpad" states. 
    t0 = "raw prompt (t0)"
    accu_t1 = "t0 + shuffled content information (t1)"
    accu_t2 = "t0 + t1 + lite CoT in [PLAN] section (t2)"
    accu_t3 = "t0 + t1 + t2 + dense CoT information in [PLAN] section (t3)"
    accu_t4 = "t0 + t1 + t2 + t3 + error-correction scratchpad (t4)"
    accu_t1_t4 = "t0 + shuffled domain information (t1) + error-correction scratchpad (t4)"
    accu_t2_t4 = "t0 + t1 + lite CoT in [PLAN] section (t2) + error-correction scratchpad (t4)"
    accu_t1_t3 = "t0 + shuffled domain information (t1) + dense CoT information in [PLAN] section (t3)"
    accu_t1_t3_t4 = "t0 + shuffled domain information (t1) + dense CoT information in [PLAN] section (t3) + error-correction scratchpad (t4)"
    t0_t5 = "raw prompt (t0) + advanced error-correction scratchpad (t5)"


PROMPT_NUM_DICT = {
    CONFIG_TYPES.t0: 1,
    CONFIG_TYPES.accu_t1: 10,
    CONFIG_TYPES.accu_t2: 10,
    CONFIG_TYPES.accu_t3: 10,
    CONFIG_TYPES.accu_t4: 30, # 10 * 3
    CONFIG_TYPES.accu_t1_t4: 30, # 10 * 3
    CONFIG_TYPES.accu_t2_t4: 30, # 10 * 3
    CONFIG_TYPES.accu_t1_t3: 10, # 10 
    CONFIG_TYPES.accu_t1_t3_t4: 30, # 10 * 3
    CONFIG_TYPES.t0_t5: 10, # 1 * 10 
}

class SPLIT_NAMES(enum.Enum):
    train = "training set"
    test_same_domain = "test set with the same domain"
    test_unseen_domain = "test set with unseen domain"
    test_longer_horizon = "test set with longer horizon"
    test_mystery_domain = "test set with mystery domain"

SUBDIRS_NAME_DICT = {  # * the subdirs name in the raw dataset, use it to access to instance pddl files
    # "zraw_data/instances/{domain_name}/{subdir_name}/instance-{instance_id}.pddl"
    SPLIT_NAMES.train.name: "generated_basic",
    SPLIT_NAMES.test_longer_horizon.name: "generated_basic_longer_plan_len",
    SPLIT_NAMES.test_same_domain.name: "generated_basic",
    SPLIT_NAMES.test_mystery_domain.name: "generated_basic",
    SPLIT_NAMES.test_unseen_domain.name: "generated_basic",
}

@enum.unique
class PLANNING_DOMAIN(enum.Enum):
    """
    PLANNING_DOMAIN is an enumeration class that represents different planning domains.
    """
    barman = "barman"
    blocksworld = "blocksworld"
    childsnack = "childsnack"
    depots = "depots"
    driverlog = "driverlog"
    grippers = "grippers"
    logistics = "logistics"
    satellite = "satellite"
    mystery_blocksworld = "mystery_blocksworld"
    obfuscated_deceptive_logistics = "obfuscated_deceptive_logistics"
    hanoi = "hanoi"
    storage = "storage"

    def get_splits(self):
        """
        Returns the splits associated with this domain.
        """
        splits_mapping = {
            "barman": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "blocksworld": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "childsnack": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "depots": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "driverlog": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "grippers": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "logistics": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "satellite": [SPLIT_NAMES.train, SPLIT_NAMES.test_same_domain, SPLIT_NAMES.test_longer_horizon],
            "mystery_blocksworld": [SPLIT_NAMES.test_mystery_domain],
            "obfuscated_deceptive_logistics": [SPLIT_NAMES.test_mystery_domain],
            "hanoi": [SPLIT_NAMES.test_unseen_domain],
            "storage": [SPLIT_NAMES.test_unseen_domain]
        }
        return splits_mapping[self.value]


class FEATURE_CLS(enum.Enum):
    domain = auto()
    instance_id = auto()
    prompt_id = auto()
    mistake_rate = auto()
    domain_text = auto()    
    problem_init_text = auto()
    problem_goal_text = auto()
    plan_text = auto()
    plan_length = auto()
    raw_plan = auto()
    full_text_prompt = auto() # domain_text + problem_init_text + problem_goal_text + plan_text # TODO check in kedro pipeline

    

def create_folder_structure(mode="default"):
    # create folder for each config, and for each config, create folders for each split
    for config_name in CONFIG_TYPES:
        config_name = config_name.name
        for split_name in SPLIT_NAMES:
            # # ! debug 
            # if split_name == SPLIT_NAMES.train:
            #     continue
            split_name = split_name.name
            folder = Path(os.path.join(
                os.path.dirname(__file__), config_name, split_name
            ))
            folder.mkdir(parents=True, exist_ok=True)
            
            if mode == "default":
                pass
            elif mode == "debug":
                # we will add a test jsonl file in each folder
                debug_dict = {
                    "domain": "blocksworld",
                    "task": "task1",
                    "plan": ["action1", "action2"],
                }
                output_lst = []
                for i in range(2):
                    temp = deepcopy(debug_dict)
                    temp.update({'id': i})
                    output_lst.append(temp)
                with jsonlines.open(os.path.join(folder, "test.jsonl"), "w") as writer:
                    writer.write_all(output_lst)
                
            elif mode == "clean":
                # we will delete test jsonl file in each folder
                for file in folder.glob("*.jsonl"):
                    file.unlink()
                folder.rmdir()
       
def test_huggingface_dataset(config_str):
    dataset_dir = os.path.dirname(__file__)
    ic(dataset_dir)
    dataset = load_dataset(dataset_dir, config_str)
    return dataset
             

if __name__ == '__main__': 
    # mode = "clean"
    # create_folder_structure(mode=mode)
    mode = "default"
    create_folder_structure(mode=mode)
    
    if mode == "debug":
        dataset = test_huggingface_dataset(CONFIG_TYPES.t0.name)