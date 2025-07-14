from __future__ import annotations
import time
import os
from learning_to_plan import config
import datasets
from learning_to_plan.data import metadata
from learning_to_plan.models.base import Model
logger = config.get_logger(__name__)
from typing import Dict, Any, Tuple


# # --- Gemini Model (Remains unchanged from previous version) ---
from google import genai
from google.genai import types

class GeminiModel(Model):
    DEFAULT_GENERATION_CONFIG = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 16384,  # 16k tokens
        "candidate_count": 1,
        "response_mime_type": "text/plain",
    }
    def __init__(self, model_name):
        super().__init__(model_name)
        assert config.GOOGLE_API_KEY, "Google API Key is required for Gemini model."
        try:
            self.client = genai.Client(
                api_key=config.GOOGLE_API_KEY,
            )

        except Exception as e:
            logger.error(f"Failed to configure Gemini model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to configure Gemini model: {e}")

    def train(self, dataset:datasets.DatasetDict, **train_kwargs) -> None: # Changed type hint
        """
        Training is not applicable for Gemini model as it is a hosted service.
        """
        logger.warning("Training is not applicable for Gemini models.")
        raise NotImplementedError("Training is not applicable for Gemini model.")

    def generate(
            self,
            chat:list[dict[str, str]],
            **generation_kwargs
        ) -> Tuple[str, dict[str, Any]]:
        logger.debug(f"Generating with Gemini model {self.model_name}.")

        gen_specs = {
            "tokens" : {}
        }

        def _map_role(role : str):
            if role == "user":
                return "user"
            elif role == "assistant":
                return "model"
            elif role == "system":
                return "model"
            else:
                raise ValueError(f"Unknown role: {role}. Expected 'user', 'assistant', or 'system'.")
        contents = []
        for msg in chat:
            contents.append(
                types.Content(
                    role=_map_role(msg["role"]),
                    parts=[
                        types.Part.from_text(text=msg["content"]),
                    ]
                )
            )
        gen_config = self.get_generation_config(**generation_kwargs)
        thinking = gen_config.pop("thinking", None)
        if thinking is not None:
            # -1 means no limit, 0 means no thinking
            thinking_budget = -1 if thinking else 0  
            thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
            gen_config["thinking_config"] = thinking_config
        config = types.GenerateContentConfig(
            **gen_config,
        )
            
        logger.debug(f"Gemini generation config: {config}")

        try:
            wait_time = generation_kwargs.get("wait_time", 20) 
            if wait_time > 0:
                logger.info(f"Waiting for {wait_time} seconds before Gemini API call.")
                time.sleep(wait_time)

            logger.info(f"Generating content with Gemini model: {self.model_name}.")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            if not response:
                raise ValueError("No response received from Gemini model.")
            
            if not response.candidates:
                raise ValueError("No candidates found in Gemini model response.")

            candidate = response.candidates[0]
            response_text = "".join(part.text for part in candidate.content.parts)

            if response.usage_metadata:
                print(response.usage_metadata)
                gen_specs["tokens"] = {
                    "input" : response.usage_metadata.prompt_token_count,
                    "output": response.usage_metadata.candidates_token_count
                }
            gen_specs["finish_reason"] = candidate.finish_reason.name

            return response_text, gen_specs
        except Exception as e:
            logger.error(f"Error generating content with Gemini model {self.model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Error generating content with Gemini model {self.model_name}: {e}") from e
        
        