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
import google.generativeai as genai

class GeminiModel(Model):
    DEFAULT_GENERATION_CONFIG = {
        "temperature": 0.7,
        "top_p": 0.93,
        "top_k": 50,
        "max_output_tokens": 2048,
        "candidate_count": 1,
        "response_mime_type": "text/plain",
        "thinking_config": {
            "thinking_budget": -1  # -1 means no limit, 0 means no thinking
        }
    }
    def __init__(self, model_name):
        super().__init__(model_name)
        assert config.GOOGLE_API_KEY, "Google API Key is required for Gemini model."
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)

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
            "tokens": {}
        }

        prompt = ""
        for msg in chat:
            prompt += f"{msg['content']}"

        # -1 means no limit, 0 means no thinking
        thinking = generation_kwargs.get("thinking", True)
        thinking_config = {
            "thinking_budget" : -1 if thinking else 0,  
        }
        generation_config = self.get_generation_config(
            **generation_kwargs,
            thinking_config=thinking_config
        )
        logger.debug(f"Gemini generation config: {generation_config}")

        try:
            wait_time = generation_kwargs.get("wait_time", 20) 
            if wait_time > 0:
                logger.info(f"Waiting for {wait_time} seconds before Gemini API call.")
                time.sleep(wait_time)

            logger.debug("Calling Gemini model.generate_content...")
            # The prompt should ideally include PLAN_START_TOKEN if Gemini needs it to trigger plan generation
            # Example: prompt_text_full = prompt_text + PLAN_START_TOKEN
            model = genai.GenerativeModel(model_name=self.model_name, generation_config=generation_config)
            token_count = model.count_tokens(prompt)
            logger.info(f"Token count for prompt: {token_count}")
            logger.info(f"Generating content with Gemini model: {self.model_name}.")
            gen_specs["tokens"]["input"] = token_count
            response = model.generate_content(prompt)
            logger.info("Gemini model generation completed successfully.")
            logger.info(f"gemini's metadata: {response.usage_metadata}")
            logger.info(f"gemini's response: {response}")
            gen_specs["tokens"]["output"] = response.usage_metadata.candidates_token_count
            try:
                text = response.text.strip()
                if not text:
                    raise ValueError("Empty response text from Gemini model.")
                return text, gen_specs
            except Exception as e:
                logger.error(f"Error extracting text from Gemini response: {e}", exc_info=True)
                raise RuntimeError(f"Error extracting text from Gemini response: {e}") from e
        except Exception as e:
            logger.error(f"Error during Gemini model generation: {e}", exc_info=True)
            raise RuntimeError(f"Error during Gemini model generation: {e}") from e