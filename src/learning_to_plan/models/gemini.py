from __future__ import annotations
import time
import os
from learning_to_plan import config
import datasets
from learning_to_plan.models.base import Model
logger = config.get_logger(__name__)


# # --- Gemini Model (Remains unchanged from previous version) ---
import google.generativeai as genai

class GeminiModel(Model):
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

    def generate_single_sample(
            self,
            chat:list[dict[str, str]],
            **generation_kwargs
        ) -> str:
        logger.debug(f"Generating with Gemini model {self.model_name}.")

        prompt = ""
        for msg in chat:
            prompt += f"{msg['content']}"
            
        generation_config = {
            "temperature":generation_kwargs.get("temperature", 0.7),
            "top_p":generation_kwargs.get("top_p", 0.93),
            "top_k":generation_kwargs.get("top_k", 50),
            "max_output_tokens":generation_kwargs.get("max_output_tokens", 2048),
            "candidate_count":1,
            "response_mime_type": "text/plain",
        }
        logger.debug(f"Gemini generation config: {generation_config}")


        # TODO: REMOVE THIS AFTER DEBUGGING
        filename = "temp__prompt__debug.txt"
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(prompt)
            logger.info(f"Prompt saved to {filename} for debugging.")

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
            response = model.generate_content(prompt)
            logger.info("Gemini model generation completed successfully.")
            logger.info(f"gemini's metadata: {response.usage_metadata}")
            logger.info(f"gemini's response: {response}")
            try:
                text = response.text.strip()
                if not text:
                    raise ValueError("Empty response text from Gemini model.")
                return text
            except Exception as e:
                logger.error(f"Error extracting text from Gemini response: {e}", exc_info=True)
                raise RuntimeError(f"Error extracting text from Gemini response: {e}") from e
        except Exception as e:
            logger.error(f"Error during Gemini model generation: {e}", exc_info=True)
            raise RuntimeError(f"Error during Gemini model generation: {e}") from e