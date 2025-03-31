"""
This is a boilerplate pipeline 'language_model_planning_evaluation'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_language_model_planning
from better_language_model_for_plan_generation.utils.constant import *
import os 
config_type_name = os.environ.get("EVAL_CONFIG_TYPE", "t0")

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_language_model_planning,
                inputs=[
                    f"params:language_model_planning_evaluation_params.{config_type_name}_params",
                    "params:language_model_planning_evaluation_general_params",
                ],
                outputs= None,
                name=f"evaluate_language_model_planning_{config_type_name}_node",
            )
        ]
    )
