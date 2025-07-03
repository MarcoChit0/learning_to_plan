
# --- get_model function (Remains unchanged) ---
from learning_to_plan.models.base import Model
from learning_to_plan.models.gemini import GeminiModel
from learning_to_plan.models.hugging_face import HuggingFaceModel
from learning_to_plan import config
logger = config.get_logger(__name__)

def get_model(model_name: str) -> Model:
    """
    Factory function to get the appropriate model based on the model name.
    """
    logger.info(f"Creating model instance for: {model_name}")
    if model_name.lower().startswith("gemini"):
        logger.info("Identified as Gemini model.")
        model_cls = GeminiModel
    else:
        logger.info("Identified as Hugging Face model.")
        model_cls = HuggingFaceModel
    try:
        model = model_cls(model_name)
    except Exception as e:
        logger.error(f"Error creating model instance: {e}", exc_info=True)
        raise e
    return model

