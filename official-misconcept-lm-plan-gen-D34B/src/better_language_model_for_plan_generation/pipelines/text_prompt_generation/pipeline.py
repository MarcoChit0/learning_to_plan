"""
This is a boilerplate pipeline 'text_prompt_generation'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import construct_huggingface_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=construct_huggingface_dataset,
            inputs=[
                'params:text_prompt_generation_params'
            ],
            outputs=None,
            name="construct_huggingface_dataset_node"
        )
    ])
