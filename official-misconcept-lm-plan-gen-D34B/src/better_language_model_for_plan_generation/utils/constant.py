import enum
from enum import IntEnum
from enum import auto

# * --- OTHER CONSTANTS --- *
# ! mistake_rate is in the config yaml file.
# * --- END OF OTHER CONSTANTS --- *

class SPECIAL_TOKENS(enum.Enum):
    PLAN = "<PLAN>"  # * plan section
    PLAN_END = "</PLAN>"
    GOAL = "<GOAL>" # * lite cot
    COUNT = "<COUNT>"
    COUNT_END = "</COUNT>"
    PRECON = "<PRECON>" # * dense cot
    PRECON_END = "</PRECON>"
    EFFECT = "<EFFECT>"
    EFFECT_END = "</EFFECT>"
    BACK = "WRONG" # * scratchpad

SPECIAL_TOKENS_DICT = { # ! DEPRECATED due to bugs https://x.com/danielhanchen/status/1781395882925343058
    "additional_special_tokens": [
        x.value for x in SPECIAL_TOKENS
    ],
}

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


PLAN_BENCH_EXAMPLE = "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]\n",


if __name__ == "__main__":
    print(list(PLANNING_DOMAIN))
    print(list(FEATURE_CLS))
