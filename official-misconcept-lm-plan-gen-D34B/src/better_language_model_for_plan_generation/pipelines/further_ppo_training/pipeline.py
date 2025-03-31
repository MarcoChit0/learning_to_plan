"""
This is a boilerplate pipeline 'further_ppo_training'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import ppo_training
import os 
from better_language_model_for_plan_generation.utils.constant import *

config_type_name = os.environ.get("EVAL_CONFIG_TYPE", "accu_t1_t4")


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=ppo_training,
            inputs=[
                f"params:further_ppo_training_params.{config_type_name}_params",
                "params:further_ppo_training_general_params"
            ],
            outputs=None,
            name=f"ppo_training_{config_type_name}_node"
        )
    ])
