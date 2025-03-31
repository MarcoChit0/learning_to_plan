"""
This is a boilerplate pipeline 'plan_validation_probing'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import probing_language_model_to_identify_mistakes
from better_language_model_for_plan_generation.utils.constant import *
import os 
config_type_name = os.environ.get("EVAL_CONFIG_TYPE", "accu_t1_t4")

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=probing_language_model_to_identify_mistakes,
            inputs=[
                "params:plan_validation_probing_params",
                f"params:plan_validation_probing_testing_model_info.{config_type_name}_params",
            ],
            outputs=None,
            name="probing_language_model_to_identify_mistakes_node",
        )
    ])
