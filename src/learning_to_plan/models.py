import datetime
import time
from typing import Optional, List, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import os
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from learning_to_plan import config
import torch
import datasets
import numpy as np

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

    def train(self, train_dataset: datasets.DatasetDict, eval_dataset: datasets.DatasetDict,  **train_kwargs) -> None: # Changed type hint
        """
        Trains the model on the provided data.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def tokenize(self, examples: List[str], max_length: int = 1024) -> Dict[str, Any]:
        """
        Tokenizes the input examples.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def decode(self, input_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes the input IDs to a string.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class HuggingFaceModel(Model):
    def __init__(self, model_name, checkpoint_dir: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        assert config.HUGGINGFACE_TOKEN, "Hugging Face token is required for model loading."

        model_source = model_name
        last_checkpoint = None
        if checkpoint_dir:
            last_checkpoint = get_last_checkpoint(checkpoint_dir)

        if last_checkpoint:
            model_source = last_checkpoint

        if last_checkpoint:
            logger.info(f"Loading model {model_name} -- checkpoint: {last_checkpoint}.")
        else:
            logger.info(f"Loading model {model_name} -- base model from Hugging Face Hub.")

        # --- Tokenizer Setup ---
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                token=config.HUGGINGFACE_TOKEN
            )

            # --- Add special tokens IF they don't exist ---
            special_tokens_to_add = []
            for separator in [config.START_OF_PLAN_TOKEN, config.END_OF_PLAN_TOKEN]:
                if separator not in self._tokenizer.get_vocab():
                    special_tokens_to_add.append(separator)

            if special_tokens_to_add:
                num_added_tokens = self._tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
                logger.info(f"Added {num_added_tokens} special tokens to tokenizer: {special_tokens_to_add}")

            logger.info(f"Tokenizer loaded. Pad token: {self._tokenizer.pad_token}, Padding side: {self._tokenizer.padding_side}")
            PLAN_START_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.START_OF_PLAN_TOKEN)
            PLAN_END_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.END_OF_PLAN_TOKEN)
            logger.info(f"Plan Start Token ID: {PLAN_START_TOKEN_ID}, Plan End Token ID: {PLAN_END_TOKEN_ID}")
            if PLAN_START_TOKEN_ID == self._tokenizer.unk_token_id or PLAN_END_TOKEN_ID == self._tokenizer.unk_token_id:
                logger.error("One or both special plan tokens were not found in the tokenizer vocabulary after attempting to add them. Check token strings and tokenizer setup.")
                raise ValueError("Special plan tokens could not be resolved to IDs.")

        except Exception as e:
            logger.error(f"Error loading or setting up tokenizer from {model_source}: {e}", exc_info=True)
            raise e

        # --- Model Loading ---
        try:
            torch_dtype = torch.bfloat16 if self.__dict__.get("bf16", False) else torch.float16

            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=config.HUGGINGFACE_TOKEN,
            )
            logger.info(f"Model loaded successfully from {model_source}.")

            # --- Resize embeddings if tokens were added ---
            if self._tokenizer.vocab_size != self._model.get_input_embeddings().weight.shape[0]:
                logger.info(f"Resizing model token embeddings to match tokenizer size: {len(self._tokenizer)}")
                self._model.resize_token_embeddings(len(self._tokenizer))
                logger.info("Model token embeddings resized successfully.")
            
            if last_checkpoint:
                logger.info(f"Loading model state from checkpoint: {last_checkpoint}")
                self._model = PeftModel.from_pretrained(
                    self._model,
                    last_checkpoint,
                    token=config.HUGGINGFACE_TOKEN,
                )
                logger.info(f"Model state loaded from checkpoint: {last_checkpoint}")

        except Exception as e:
            logger.error(f"Error loading model from {model_source} or resizing embeddings: {e}", exc_info=True)
            raise e

    def decode(self, input_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(
            input_ids,
            skip_special_tokens=skip_special_tokens
        )

    def tokenize(self, examples: List[str], max_length: int = 1024) -> Dict[str, Any]:
            PLAN_START_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.START_OF_PLAN_TOKEN)
            PLAN_END_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.END_OF_PLAN_TOKEN)
            if PLAN_START_TOKEN_ID == self._tokenizer.unk_token_id or PLAN_END_TOKEN_ID == self._tokenizer.unk_token_id:
                raise ValueError("Special plan tokens could not be resolved to IDs. Check token strings and tokenizer setup.")

            batch_input_ids = []
            batch_labels = []
            batch_attention_mask = []

            for text_instance in examples["text"]:
                tokenized_output = self._tokenizer(
                    text_instance + self._tokenizer.eos_token, # Add EOS token at the end
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors=None,
                    add_special_tokens=False, # We are handling special plan tokens manually
                    return_attention_mask=True
                )

                input_ids_padded = tokenized_output["input_ids"]
                attention_mask_padded = tokenized_output["attention_mask"]

                try:
                    plan_start_idx = input_ids_padded.index(PLAN_START_TOKEN_ID)
                except ValueError:
                    # Log more context for debugging
                    logger.error(f"PLAN_START_TOKEN_ID {PLAN_START_TOKEN_ID} ('{config.START_OF_PLAN_TOKEN}') not found in input_ids. Input text (first 200 chars): {text_instance[:200]}...")
                    logger.error(f"Input IDs (first 20): {input_ids_padded[:20]}")
                    raise ValueError(f"PLAN_START_TOKEN_ID {PLAN_START_TOKEN_ID} not found.")

                try:
                    # Search for plan_end_idx *after* plan_start_idx
                    plan_end_idx = input_ids_padded.index(PLAN_END_TOKEN_ID, plan_start_idx)
                except ValueError:
                    logger.error(f"PLAN_END_TOKEN_ID {PLAN_END_TOKEN_ID} ('{config.END_OF_PLAN_TOKEN}') not found after plan_start_idx in input_ids. Input text (first 200 chars): {text_instance[:200]}...")
                    logger.error(f"Input IDs (segment around plan_start_idx): {input_ids_padded[max(0, plan_start_idx-10):plan_start_idx+20]}")
                    raise ValueError(f"PLAN_END_TOKEN_ID {PLAN_END_TOKEN_ID} not found after plan_start_idx.")

                labels = [-100] * len(input_ids_padded)
                # Labels include the plan_start and plan_end tokens
                labels[plan_start_idx : plan_end_idx + 1] = input_ids_padded[plan_start_idx : plan_end_idx + 1]

                batch_input_ids.append(input_ids_padded)
                batch_labels.append(labels)
                batch_attention_mask.append(attention_mask_padded)

            return {
                "input_ids": batch_input_ids,
                "labels": batch_labels,
                "attention_mask": batch_attention_mask,
            }

    def train(self, checkpoint_dir: str, tokenized_train_dataset: datasets.DatasetDict, tokenized_eval_dataset: datasets.DatasetDict, **train_kwargs: Dict[str, Any]) -> None:

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        start_timer = datetime.datetime.now()
        logger.info(f"Training started at {start_timer}.")

        # --- LoRA Setup ---
        if not isinstance(self._model, PeftModel):
            logger.info("Applying LoRA configuration to the base model.")
            lora_r = train_kwargs.get("lora_r", 512) # Default LoRA r
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=train_kwargs.get("lora_alpha", lora_r * 2), # Default LoRA alpha
                target_modules=train_kwargs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]),
                lora_dropout=train_kwargs.get("lora_dropout", 0.05), # Default LoRA dropout
                bias=train_kwargs.get("lora_bias", "none"), # Default LoRA bias
                task_type="CAUSAL_LM",
            )
            self._model = get_peft_model(self._model, lora_cfg)
        else:
            logger.info("Model is already a PeftModel (likely loaded from checkpoint). Skipping LoRA application.")
        self._model.print_trainable_parameters()

        collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        # --- Print GPU Information ---
        logger.info(f"PyTorch version: {torch.__version__}")
        is_cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {is_cuda_available}")

        if not is_cuda_available:
            raise RuntimeError("CUDA is not available. Training requires a GPU.")

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu_stats.name}, Total Memory: {max_memory} GB")
        logger.info(f"Initial Reserved Memory: {start_gpu_memory} GB")
            

        # --- Training Arguments ---
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
            # load_best_model_at_end=train_kwargs.get("load_best_model_at_end", True),
            # eval_on_start=train_kwargs.get("eval_on_start", False),
        )
        logger.debug(f"Final TrainingArguments: {training_args.to_dict()}")

        # --- Trainer Initialization ---
        trainer = Trainer(
            model=self._model,
            args=training_args,
            data_collator=collator,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
        )

        # --- Start Training ---
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        resume_from_checkpoint = last_checkpoint if last_checkpoint else None
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("No valid checkpoint found in output directory. Starting training from scratch or loaded model.")
        
        try:
            logger.info("Starting trainer.train()...")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            logger.info("trainer.train() finished.")
        except Exception as e:
            logger.error(f"Error during trainer.train(): {e}", exc_info=True) # Log full traceback
            raise ValueError(f"Error during training: {e}") from e # Re-raise with context
        
        try:
            logger.info(f"Saving final model and trainer state to {checkpoint_dir}")
            # Ensure the PeftModel is saved correctly. trainer.save_model() handles this for PeftModel.
            self._model.save_pretrained(checkpoint_dir) # Recommended for PEFT models
            self._tokenizer.save_pretrained(checkpoint_dir) # Save tokenizer explicitly
            # trainer.save_model(checkpoint_dir) # This also works and saves adapter for PEFT
            trainer.save_state()
            logger.info(f"Model, tokenizer and trainer state saved successfully to {checkpoint_dir}.")
        except Exception as e:
            logger.error(f"Error saving model and state: {e}", exc_info=True)
            raise e # Re-raise after logging
        
        try:
            metrics = train_result.metrics
            metrics["train_samples"] = len(tokenized_train_dataset) # Add number of train samples
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics) # Save metrics to a file

            end_timer = datetime.datetime.now()
            logger.info(f"Training completed at {end_timer}. Duration: {end_timer - start_timer}")
            
            if is_cuda_available: # Log memory usage only if CUDA was used
                end_gpu_memory = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024, 3) 
                used_memory_for_train = round(end_gpu_memory - start_gpu_memory, 3)
                logger.info(f"Max Reserved Memory during training: {end_gpu_memory} GB.")
                if max_memory > 0: # Avoid division by zero if max_memory wasn't set (e.g., error getting props)
                    used_memory_ratio = round(end_gpu_memory / max_memory, 3)
                    logger.info(f"Max Reserved Memory Ratio: {used_memory_ratio*100:.2f}%")
                    logger.info(f"Memory increase during training: {used_memory_for_train} GB.")
            try:
                # Calculate checkpoint size robustly
                checkpoint_size_gb = 0
                if os.path.exists(checkpoint_dir):
                    checkpoint_size_bytes = sum(
                        os.path.getsize(os.path.join(root, f))
                        for root, _, files in os.walk(checkpoint_dir)
                        for f in files
                    )
                    checkpoint_size_gb = checkpoint_size_bytes / (1024 ** 3)
                logger.info(f"Final checkpoint size: {checkpoint_size_gb:.3f} GB at {checkpoint_dir}.")
            except Exception as size_e:
                logger.warning(f"Could not calculate checkpoint size: {size_e}")

        except Exception as e: # Catch errors in metrics logging as well
            logger.error(f"Error during training metrics logging or final summary: {e}", exc_info=True)
            # Don't re-raise here if training itself was successful.

    def generate(self, prompt: str, **generation_kwargs) -> list[str]:
        """
        Generates text based on a prompt using the Hugging Face model.

        Parameters:
            prompt: The input prompt text (should end before <|plan_start|>).
            **generation_kwargs: Keyword arguments for generation control (e.g., max_new_tokens, temperature).

        Returns:
            A list containing the generated plan(s), stopping at <|plan_end|> or max_new_tokens.
        """
        logger.debug("Generating with Hugging Face model.")

        # --- Generation Configuration ---
        gen_kwargs = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 512),
            "do_sample": generation_kwargs.get("do_sample", True),
            "temperature": generation_kwargs.get("temperature", 0.7),
            "top_p": generation_kwargs.get("top_p", 0.93),
            "top_k": generation_kwargs.get("top_k", 50),
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
            "num_return_sequences": generation_kwargs.get("num_return_sequences", 1),
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        logger.debug(f"Generation parameters: {gen_kwargs}")

        # Ensure model is in evaluation mode
        self._model.eval()

        device = next(self._model.parameters()).device

        inputs = self._tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
            max_length=generation_kwargs.get("max_seq_length", 512),
            return_tensors="pt",
        ).to(device)

        input_length = inputs.input_ids.shape[1]

        # --- Perform Generation ---
        with torch.no_grad():
            dtype = getattr(self._model, 'dtype', torch.float16)
            use_autocast = (device.type == 'cuda') and dtype in [torch.float16, torch.bfloat16]
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                outputs = self._model.generate(
                    **inputs,
                    **gen_kwargs
                )
        
        # --- print input & output tokens ids and respective text ---
        logger.debug(f"Input IDs: {inputs.input_ids}...")
        logger.debug(f"Input Text: {self.decode(inputs.input_ids[0], skip_special_tokens=False)}...")  
        logger.debug(f"Input Length: {inputs.input_ids.shape[1]} tokens.")

        logger.debug(f"Output IDs: {outputs}...")
        logger.debug(f"Output Text: {self.decode(outputs[0], skip_special_tokens=False)}...")
        logger.debug(f"Output Length: {outputs.shape[1]} tokens.")
        logger.debug(f"Output Text (after input length): {self.decode(outputs[0][input_length:], skip_special_tokens=False)}...")
        # --- Process Outputs ---
        generated_texts = []
        for output_sequence in outputs:
            generated_tokens = (output_sequence[input_length:] if output_sequence.shape[0] > input_length else torch.tensor([], dtype=torch.long, device=device))
            
            # --- Identify the indices for plan boundaries ---
            plan_start_idx = next((i for i, token in enumerate(generated_tokens) if token == self._tokenizer.convert_tokens_to_ids(config.START_OF_PLAN_TOKEN)), None)
            if plan_start_idx is None:
                generated_texts.append(f"Generation Error: {config.START_OF_PLAN_TOKEN} not found.")
                continue

            plan_end_idx = next(
            (i for i, token in enumerate(generated_tokens) if token == self._tokenizer.convert_tokens_to_ids(config.END_OF_PLAN_TOKEN)),None,)
            if plan_end_idx is None:
                generated_texts.append(f"Generation Error: {config.END_OF_PLAN_TOKEN} not found.")
                continue

            # --- Remove everything between (and including) the special tokens ---
            remaining_tokens = torch.cat((generated_tokens[:plan_start_idx], generated_tokens[plan_end_idx + 1:]))
            if remaining_tokens.shape[0] == 0:
                generated_texts.append("Generation Error: No text left after removing plan.")
                continue

            decoded_text = self.decode(remaining_tokens, skip_special_tokens=True)
            generated_texts.append(decoded_text.strip())
        
        if not generated_texts:
            raise RuntimeError("No valid generated texts found.")

        return generated_texts


# # --- Gemini Model (Remains unchanged from previous version) ---
# import google.generativeai as genai
# class GeminiModel(Model):
#     def __init__(self, model_name, **kwargs):
#         super().__init__(model_name, **kwargs)
#         assert config.GOOGLE_API_KEY, "Google API Key is required for Gemini model."
#         try:
#             genai.configure(api_key=config.GOOGLE_API_KEY)
#             logger.info("Gemini API configured successfully.")
#         except Exception as e:
#             logger.error(f"Failed to configure Gemini model: {e}", exc_info=True)
#             raise RuntimeError(f"Failed to configure Gemini model: {e}")

#     def train(self, dataset:datasets.DatasetDict, **train_kwargs) -> None: # Changed type hint
#         """
#         Training is not applicable for Gemini model as it is a hosted service.
#         """
#         logger.warning("Training is not applicable for Gemini models.")
#         raise NotImplementedError("Training is not applicable for Gemini model.")

#     def generate(
#             self,
#             prompt_text: str,
#             **generation_kwargs
#         ) -> list[str]:
#         logger.debug(f"Generating with Gemini model {self._model_name}.")

#         generation_config = genai.types.GenerationConfig( # Use GenerationConfig object
#             temperature=generation_kwargs.get("temperature", 0.7),
#             top_p=generation_kwargs.get("top_p", 0.93),
#             top_k=generation_kwargs.get("top_k", 50),
#             max_output_tokens=generation_kwargs.get("max_new_tokens", 2048),
#             candidate_count=generation_kwargs.get("num_return_sequences", 1), # Map num_return_sequences
#             stop_sequences=[PLAN_END_TOKEN] # Add plan end token as stop sequence
#         )
#         logger.debug(f"Gemini generation config: {generation_config}")


#         try:
#             model = genai.GenerativeModel(
#                 self._model_name,
#                 generation_config=generation_config,
#                 # safety_settings= # Add safety settings if needed
#             )
#         except Exception as e:
#             logger.error(f"Failed to initialize Gemini model '{self._model_name}': {e}", exc_info=True)
#             raise RuntimeError(f"Failed to initialize Gemini model '{self._model_name}': {e}") from e

#         try:
#             wait_time = generation_kwargs.get("wait_time", 0) # Default to 0 wait time unless specified
#             if wait_time > 0:
#                 logger.info(f"Waiting for {wait_time} seconds before Gemini API call.")
#                 time.sleep(wait_time)

#             logger.debug("Calling Gemini model.generate_content...")
#             # The prompt should ideally include PLAN_START_TOKEN if Gemini needs it to trigger plan generation
#             # Example: prompt_text_full = prompt_text + PLAN_START_TOKEN
#             response = model.generate_content(prompt_text) # Use original prompt_text for now
#             logger.debug("Gemini API call completed.")


#             generated_texts = []
#             if response and response.candidates:
#                 for candidate in response.candidates:
#                     if candidate.content and candidate.content.parts:
#                         text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
#                         # Remove potential trailing PLAN_END_TOKEN if stop sequence worked
#                         if text.endswith(PLAN_END_TOKEN):
#                              text = text[:-len(PLAN_END_TOKEN)]
#                         generated_texts.append(text.strip())
#                     elif candidate.content and not candidate.content.parts:
#                          logger.warning(f"Gemini candidate content has no parts: {candidate.content}")
#                     else:
#                          logger.warning(f"Gemini candidate has no content: {candidate}")

#             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
#                  logger.error(f"Gemini request blocked. Reason: {response.prompt_feedback.block_reason}")
#                  raise RuntimeError(f"Gemini request blocked. Reason: {response.prompt_feedback.block_reason}")
#             if not generated_texts and response.candidates:
#                  finish_reasons = [c.finish_reason for c in response.candidates]
#                  logger.warning(f"No text extracted from Gemini response. Finish reasons: {finish_reasons}")


#             if not generated_texts:
#                 logger.error(f"No valid generated texts found in Gemini response. Response: {response}")
#                 raise RuntimeError("No valid generated texts found in Gemini response.")

#             logger.debug(f"Generated {len(generated_texts)} sequences from Gemini.")
#             return generated_texts

#         except Exception as e:
#             logger.error(f"Failed to generate text with Gemini model '{self._model_name}': {e}", exc_info=True)
#             raise RuntimeError(f"Failed to generate text with Gemini model '{self._model_name}': {e}") from e

# --- get_model function (Remains unchanged) ---
def get_model(model_name: str, **kwargs) -> Model:
    """
    Factory function to get the appropriate model based on the model name.
    """
    # logger.info(f"Creating model instance for: {model_name}")
    # if model_name.lower().startswith("gemini"):
    #     logger.info("Identified as Gemini model.")
    #     return GeminiModel(model_name, **kwargs)
    # else:
    logger.info("Identified as Hugging Face model.")
    checkpoint_dir = kwargs.pop('checkpoint_dir', None)
    return HuggingFaceModel(model_name, checkpoint_dir=checkpoint_dir, **kwargs)
