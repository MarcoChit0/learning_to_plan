# Import necessary HF classes
import datetime
import time
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import os
from transformers.trainer_utils import get_last_checkpoint
# Import PEFT components
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from learning_to_plan import config
from learning_to_plan import task
import torch
import datasets

logger = config.get_logger(__name__)

class Model:
    def __init__(self, model_name, **kwargs):
        self._model_name = model_name
        self.__dict__.update(kwargs)

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generates a plan based on the provided prompt.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def train(self, dataset:datasets.Dataset, **train_kwargs) -> None:
        """
        Trains the model on the provided data.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class HuggingFaceModel(Model):
    def __init__(self, model_name, checkpoint_dir: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        assert config.HUGGINGFACE_TOKEN, "Hugging Face token is required for model loading."

        model_source = model_name
        if checkpoint_dir:
            last_checkpoint = None
            last_checkpoint = get_last_checkpoint(checkpoint_dir)

            if last_checkpoint:
                model_source = last_checkpoint

        logger.info(f"Determined model source: {model_source} ({'Checkpoint' if last_checkpoint else 'Base Model'})")

        assert config.HUGGINGFACE_TOKEN, "Hugging Face token is required for model loading."
        self._tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, token=config.HUGGINGFACE_TOKEN)
        # if get_config("load_in_8bit"):
        #     model = AutoModelForCausalLM.from_pretrained(
        #         model_source,
        #         trust_remote_code=True,
        #         device_map="auto",
        #         quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        #         token=HUGGINGFACE_TOKEN,
        #     )
        #     lora_r = get_config("lora_r", 8)
        #     # attach LoRA adapter so the model becomes trainable
        #     lora_cfg = LoraConfig(
        #         r=lora_r,
        #         lora_alpha=get_config("lora_alpha", lora_r*4),
        #         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
        #                         "up_proj", "down_proj", "gate_proj"],
        #         lora_dropout=get_config("lora_dropout", 0.05),
        #         bias=get_config("lora_bias", "none"),
        #         task_type="CAUSAL_LM",
        #     )
        #     model = get_peft_model(model, lora_cfg)
        #     model.print_trainable_parameters()
        self._model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.__dict__.get("bf16", None) else torch.float16,
                token=config.HUGGINGFACE_TOKEN,
            )
    
    # --- Training Arguments ---
    # training_args = TrainingArguments(
    #     output_dir=model_checkpoint_dir,
    #     run_name=f"{config.get_config('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #     report_to=config.get_config("report_to", "none"),
    #     num_train_epochs=config.get_config("num_train_epochs", 2),
    #     per_device_train_batch_size=config.get_config("batch_size", 1),
    #     per_device_eval_batch_size=config.get_config("per_device_eval_batch_size", 1),
    #     gradient_accumulation_steps=config.get_config("gradient_accumulation_steps", 1),
    #     fp16=not config.get_config("bf16", False),
    #     bf16=config.get_config("bf16", False),
    #     learning_rate=config.get_config("learning_rate", 1.0e-5),
    #     lr_scheduler_type=config.get_config("lr_scheduler_type", "cosine"),
    #     weight_decay=config.get_config("weight_decay", 0.02),
    #     save_strategy=config.get_config("save_strategy", "steps"),
    #     save_steps=config.get_config("save_steps", 800),
    #     save_total_limit=config.get_config("save_total_limit", 1),
    #     logging_strategy=config.get_config("logging_strategy", "steps"),
    #     logging_steps=config.get_config("logging_steps", 400),
    #     eval_strategy=config.get_config("eval_strategy", "epoch"),
    #     optim=config.get_config("optimizer", "adamw_8bit"),
    # )
    def train(self, checkpoint_dir, dataset:datasets.Dataset, **train_kwargs) -> None:
        assert "train" in dataset, "Training dataset is required for training."
        assert "validation" in dataset, "Validation dataset is required for training."
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        start_timer = datetime.datetime.now()
        logger.info(f"Training started at {start_timer}.")

        def tokenize_fn(examples):
            return self._tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=train_kwargs.get("max_length", 512)
            )

        try:
            train_dataset = dataset['train'].map(
                tokenize_fn,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing training dataset",
            )
            eval_dataset = dataset['validation'].map(
                tokenize_fn,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing validation dataset",
            )
        except Exception as e:
            logger.error(f"Error during dataset tokenization: {e}")
            raise e

        collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            run_name=f"{self._model_name}-{os.path.basename(checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-debug",
            report_to=train_kwargs.get("report_to", "none"),
            num_train_epochs=train_kwargs.get("num_train_epochs", 2),
            per_device_train_batch_size=train_kwargs.get("batch_size", 1),
            gradient_accumulation_steps=train_kwargs.get("gradient_accumulation_steps", 1),
            fp16=not self.__dict__.get("bf16", False), 
            bf16=self.__dict__.get("bf16", False),     
            learning_rate=train_kwargs.get("learning_rate", 1.0e-5),
            lr_scheduler_type=train_kwargs.get("lr_scheduler_type", "cosine"),
            weight_decay=train_kwargs.get("weight_decay", 0.02),
            save_strategy=train_kwargs.get("save_strategy", "epoch"),
            save_total_limit=train_kwargs.get("save_total_limit", 1),
            logging_strategy=train_kwargs.get("logging_strategy", "epoch"),
            eval_strategy=train_kwargs.get("eval_strategy", "epoch"),
            optim=train_kwargs.get("optimizer", "adamw_8bit"),
        )

        logger.info(f"Training arguments: {training_args}")
        try:
            trainer = Trainer(
                model=self._model,
                args=training_args,
                data_collator=collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            trainer.train()
            trainer.save_model(checkpoint_dir)
            end_timer = datetime.datetime.now()
            logger.info(f"Training completed at {end_timer}. Duration: {end_timer - start_timer}")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise e

    def generate(self, prompt: str, **generation_kwargs) -> list[str]:
        """
        Generates text based on a prompt using the Hugging Face model.

        Parameters:
            prompt: The input prompt text.
            **gen_kwargs_input: Keyword arguments for generation control (e.g., max_new_tokens, temperature).

        Returns:
            The generated text, excluding the prompt.
        """
        logger.debug("Generating with Hugging Face model.")

        if not hasattr(self, '_model') or not hasattr(self, '_tokenizer'):
             raise RuntimeError("Model or tokenizer not initialized.")

        device = next(self._model.parameters()).device
        dtype = self._model.dtype # Get model's dtype directly

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=generation_kwargs.get("max_length", 512),
        ).to(device)
        input_length = inputs.input_ids.shape[1]

        # Build generation kwargs dictionary using values from gen_kwargs_input or defaults
        gen_kwargs = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 512),
            "do_sample": generation_kwargs.get("do_sample", True),
            "temperature": generation_kwargs.get("temperature", 0.7),
            "top_p": generation_kwargs.get("top_p", 0.93),
            "top_k": generation_kwargs.get("top_k", 50),
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.eos_token_id, # Use EOS for padding during generation
            "num_return_sequences": 1, # Required for single string output
        }
        # Filter out None values if defaults could be None and shouldn't be passed
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # Check for quantization
        is_quantized = getattr(self._model, 'is_loaded_in_8bit', False) 
        use_autocast = (device.type == 'cuda') and not is_quantized

        with torch.no_grad():
            # Use autocast context manager for mixed precision if applicable
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                outputs = self._model.generate(
                    **inputs,
                    **gen_kwargs
                )

        generated_texts = []
        input_length = inputs.input_ids.shape[1]
        for output_sequence in outputs:
            generated_tokens = output_sequence[input_length:] if output_sequence.shape[0] > input_length else torch.tensor([], dtype=torch.long, device=device)
            decoded_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(decoded_text.strip())

        if not generated_texts:
            raise RuntimeError("No valid generated texts found.")

        return generated_texts


import google.generativeai as genai
class GeminiModel(Model):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        assert config.GOOGLE_API_KEY, "Google API Key is required for Gemini model."
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to configure Gemini model: {e}")
        
    def train(self, dataset:datasets.Dataset, **train_kwargs) -> None:
        """
        Training is not applicable for Gemini model as it is a hosted service.
        """
        raise NotImplementedError("Training is not applicable for Gemini model.")

    def generate(
            self,
            prompt_text: str,
            **generation_kwargs
        ) -> list[str]:
        logger.debug(f"Generating with Gemini model {self._model_name}.")

        generation_config = {
            "temperature": generation_kwargs.get("temperature", 0.7),
            "top_p": generation_kwargs.get("top_p", 0.93), # Ensure top_p is included
            "top_k": generation_kwargs.get("top_k", 50), # Optional: top-k sampling
            "max_output_tokens": generation_kwargs.get("max_new_tokens", 2048), # Optional: max output tokens
            "candidate_count": generation_kwargs.get("candidate_count", 1), # Optional: number of candidates
        }

        try:
            model = genai.GenerativeModel(
                self._model_name,
                generation_config=generation_config,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model '{self._model_name}': {e}") from e

        try:
            # Gemini free tier accepts 4 requests per minute. Introducing a wait time of 20 seconds to avoid throttling.
            wait_time = generation_kwargs.get("wait_time", 20)
            if wait_time > 0:
                logger.info(f"Waiting for {wait_time} seconds to avoid throttling.")
                time.sleep(wait_time)
            generated_texts = []
            response = model.generate(prompt_text)
            if response and hasattr(response, 'candidates'):
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts: 
                        text = ""
                        for part in candidate.content.parts:
                            text += part.text
                        generated_texts.append(text.strip())
                    else:
                        raise RuntimeError(f"Candidate content is empty or malformed: {candidate}.")
            else:
                raise RuntimeError(f"Response from Gemini model is empty or malformed: {response}.")
            if not generated_texts:
                raise RuntimeError("No valid generated texts found.")
            return generated_texts
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Gemini model '{self._model_name}': {e}") from e
        
def get_model(model_name: str, **kwargs) -> Model:
    """
    Factory function to get the appropriate model based on the model name.
    """
    if model_name.lower().startswith("gemini"):
        return GeminiModel(model_name, **kwargs)
    else:
        return HuggingFaceModel(model_name, **kwargs)
    