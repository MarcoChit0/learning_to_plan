"""
This is a boilerplate pipeline 'training_llm_for_plan_generation'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import model_training


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_training,
                inputs=[
                    "params:training_llm_for_plan_generation_params"
                ],
                outputs=None,  # actual output will be the checkpoint of the trained LLM
                name="model_training_node",
            )
        ]
    )
