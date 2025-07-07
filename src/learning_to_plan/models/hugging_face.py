from __future__ import annotations
import datetime
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
from learning_to_plan.models.base import Model

logger = config.get_logger(__name__)

class HuggingFaceModel(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        assert config.HUGGINGFACE_TOKEN, "Hugging Face token is required for model loading."

    def setup(self, prompt_type: config.PROMPT_TYPE, checkpoint_dir: Optional[str] = None, is_trainable=False,  **kwargs) -> None:
        model_source = self.model_name
        last_checkpoint = None
        if checkpoint_dir:
            last_checkpoint = get_last_checkpoint(checkpoint_dir)

        if last_checkpoint:
            model_source = last_checkpoint

        if last_checkpoint:
            logger.info(f"Loading model {self.model_name} -- checkpoint: {last_checkpoint}.")
        else:
            logger.info(f"Loading model {self.model_name} -- base model from Hugging Face Hub.")

        self.metadata["checkpoint"] = last_checkpoint
        # --- Tokenizer Setup ---
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            token=config.HUGGINGFACE_TOKEN
            )

            # --- Add special tokens IF they don't exist ---
            special_tokens_to_add = []

            for tok in config.get_special_tokens(prompt_type=prompt_type):
                if tok not in self.tokenizer.get_vocab():
                    special_tokens_to_add.append(tok)

            if special_tokens_to_add:
                num_added_tokens = self.tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
                assert num_added_tokens == len(special_tokens_to_add), f"Expected to add {len(special_tokens_to_add)} special tokens, but added {num_added_tokens}."
                logger.info(f"Added {num_added_tokens} special tokens to tokenizer: {special_tokens_to_add}")

                for token in special_tokens_to_add:
                    if token not in self.tokenizer.get_vocab():
                        raise ValueError(f"Special token '{token}' not found in tokenizer vocabulary. It will be added.")
                    else:
                        logger.info(f"Special token '{token}' successfully added to tokenizer vocabulary with ID {self.tokenizer.convert_tokens_to_ids(token)}.")
            self.metadata['vocab_size'] = self.tokenizer.vocab_size
        except Exception as e:
            logger.error(f"Error loading or setting up tokenizer from {model_source}: {e}", exc_info=True)
            raise e

        # --- Model Loading ---
        try:
            torch_dtype = torch.bfloat16 if kwargs.get("bf16", False) else torch.float16
            self.metadata['torch_dtype'] = str(torch_dtype)

            quantization_config_param = None
            if  kwargs.get("load_in_8bit", False): # Check if load_in_8bit is true from train_config.json
                quantization_config_param = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("8-bit quantization enabled for model loading.")
                self.metadata['quantization_config'] = quantization_config_param.to_dict()

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=config.HUGGINGFACE_TOKEN,
                device_map="auto",
                quantization_config=quantization_config_param,
            )
            logger.info(f"Model loaded successfully from {model_source}.")

            # --- Resize embeddings if tokens were added ---
            if self.tokenizer.vocab_size != self.model.get_input_embeddings().weight.shape[0]:
                logger.info(f"Resizing model token embeddings to match tokenizer size: {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info("Model token embeddings resized successfully.")
                # ---- Initialize new token embeddings with almost zero values ----
                if not last_checkpoint and special_tokens_to_add:
                    embedding_layer = self.model.get_input_embeddings()
                    reference_token = "<|endoftext|>"
                    reference_token_id = self.tokenizer.convert_tokens_to_ids(reference_token)
                    logger.info(
                        msg=f"Token {', '.join(special_tokens_to_add)} cannot initialize its embedding layer with almost zero values. "
                        f"Setting it to the same as the reference token ID: {reference_token}, {reference_token_id}."
                    )
                    with torch.no_grad():
                        for token in special_tokens_to_add:
                            token_id = self.tokenizer.convert_tokens_to_ids(token)
                            embedding_layer.weight[token_id].copy_(embedding_layer.weight[reference_token_id])

            if quantization_config_param and is_trainable:
                logger.info("Preparing model for 8-bit training.")
                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
                logger.info("Model prepared for 8-bit training.")
            
            if last_checkpoint:
                logger.info(f"Loading model state from checkpoint: {last_checkpoint}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    last_checkpoint,
                    token=config.HUGGINGFACE_TOKEN,
                    is_trainable=is_trainable,
                )
                logger.info(f"Model state loaded from checkpoint: {last_checkpoint}")

        except Exception as e:
            logger.error(f"Error loading model from {model_source} or resizing embeddings: {e}", exc_info=True)
            raise e

    def get_token_probability(self, input_tokens, target_token):
        with torch.no_grad():
            outputs = self.model(input_tokens)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        token_prob = probs[0, target_token]
        return token_prob


    def get_token_probability(self, input_tokens, target_token):
        with torch.no_grad():
            outputs = self.model(input_tokens)
        # get the logits for our model output
        logits = outputs.logits[:, -1, :]
        # calculate the softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        token_prob = probs[0, target_token]
        return token_prob


    def decode(self, input_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(
            input_ids,
            skip_special_tokens=skip_special_tokens
        )

    def tokenize_chat(self, chat:list[dict[str, str]], max_seq_length: int = 1024) -> Dict[str, Any]:
        """
        Tokenizes a single chat conversation for training.
        """
        # --- Tokenize the chat conversation ---
        tokenized_chat = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_dict=True
        )
        input_ids = tokenized_chat["input_ids"][0].tolist()
        attention_mask = tokenized_chat["attention_mask"][0].tolist()

        labels = input_ids.copy()
        for i in range(len(labels)):
            if labels[i] == self.tokenizer.pad_token_id:
                labels[i] = -100



        # --- verify the plan ---
        # The plan is between the start and end tokens (inclusive)
        PLAN_START_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(config.TOKENS.PLAN_START.value)
        PLAN_END_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(config.TOKENS.PLAN_END.value)
        plan_start_index = next((i for i, x in enumerate(input_ids) if x == PLAN_START_TOKEN_ID), None)
        if plan_start_index is None:
            raise ValueError(f"Plan start token not found in response input IDs for the chat {chat}.")
        
        plan_end_index = next((i for i, x in enumerate(input_ids) if x == PLAN_END_TOKEN_ID), None)
        if plan_end_index is None:
            raise ValueError(f"Plan end token not found in response input IDs for the chat {chat}.")
        
        assert plan_end_index > plan_start_index, f"Plan end token index {plan_end_index} must be greater than start token index {plan_start_index}."
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def train(self, checkpoint_dir: str, tokenized_train_dataset: datasets.DatasetDict, tokenized_eval_dataset: datasets.DatasetDict, **train_kwargs: Dict[str, Any]) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        start_timer = datetime.datetime.now()
        logger.info(f"Training started at {start_timer}.")

        # --- LoRA Setup ---
        if not isinstance(self.model, PeftModel):
            logger.info("Applying LoRA configuration to the base model.")
            lora_r = train_kwargs.get("lora_r", 512) # Default LoRA r
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=train_kwargs.get("lora_alpha", lora_r * 2), # Default LoRA alpha
                lora_dropout=train_kwargs.get("lora_dropout", 0.1),
                bias=train_kwargs.get("lora_bias", "none"),
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
        else:
            logger.info("Model is already a PeftModel (likely loaded from checkpoint). Skipping LoRA application.")
        self.model.print_trainable_parameters()

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # --- Print GPU Information ---
        logger.info(f"PyTorch version: {torch._version__}")
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
            run_name=f"{self.model_name}-{os.path.basename(checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-debug",
            report_to=train_kwargs.get("report_to", "none"),
            num_train_epochs=train_kwargs.get("num_train_epochs", 2),
            per_device_train_batch_size=train_kwargs.get("batch_size", 1),
            gradient_accumulation_steps=train_kwargs.get("gradient_accumulation_steps", 1),
            fp16=not self._dict__.get("bf16", False), 
            bf16=self._dict__.get("bf16", False),     
            learning_rate=train_kwargs.get("learning_rate", 1.0e-5),
            lr_scheduler_type=train_kwargs.get("lr_scheduler_type", "cosine"),
            weight_decay=train_kwargs.get("weight_decay", 0.02),
            save_strategy=train_kwargs.get("save_strategy", "epoch"),
            save_total_limit=train_kwargs.get("save_total_limit", 1),
            logging_strategy=train_kwargs.get("logging_strategy", "epoch"),
            eval_strategy=train_kwargs.get("eval_strategy", "epoch"),
            optim=train_kwargs.get("optimizer", "adamw_8bit"),
            warmup_ratio=train_kwargs.get("warmup_ratio", 0.1),
            label_names=["labels"],
        )
        logger.debug(f"Final TrainingArguments: {training_args.to_dict()}")

        # --- Trainer Initialization ---
        trainer = Trainer(
            model=self.model,
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
            self.model.save_pretrained(checkpoint_dir, save_embedding_layers=True) # Recommended for PEFT models
            self.tokenizer.save_pretrained(checkpoint_dir) # Save tokenizer explicitly
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

    def generate_single_sample(self, chat:list[dict[str, str]], **generation_kwargs) -> str:
        """
        Generates text based on a prompt using the Hugging Face model.

        The model now continues the last answer that was started with "My plan is as follows:" 
        and the special start-of-plan token.
        Parameters:
            prompt: The input prompt text (should end before <|plan_start|>).
            **generation_kwargs: Keyword arguments for generation control (e.g., max_new_tokens, temperature).
        Returns:
            A list containing the generated plan(s), stopping at <|plan_end|> or max_new_tokens.
        """
        logger.debug("Generating with Hugging Face model.")

        # Ensure model is in evaluation mode
        self.model.eval()

        device = next(self.model.parameters()).device

        # --- Generation Configuration ---
        gen_kwargs = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 512),
            "do_sample": generation_kwargs.get("do_sample", True),
            "temperature": generation_kwargs.get("temperature", 0.7),
            "top_p": generation_kwargs.get("top_p", 0.93),
            "top_k": generation_kwargs.get("top_k", 50),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": 1,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        logger.debug(f"Generation parameters: {gen_kwargs}")
        try:
            # Set add_generation_prompt=False so the model continues from the provided assistant prompt
            inputs = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                padding=False,              
                truncation=True,            
                max_length=generation_kwargs.get("max_prompt_length", 2048),
                return_tensors="pt",
                return_attention_mask=True,
                return_dict=True,
            ).to(device)
        except Exception as e:
            raise ValueError(f"Error during tokenizer.apply_chat_template: {e}") from e

        input_length = inputs["input_ids"].shape[1]

        # --- Perform Generation ---
        with torch.no_grad():
            dtype = getattr(self.model, 'dtype', torch.float16)
            use_autocast = (device.type == 'cuda') and dtype in [torch.float16, torch.bfloat16]
            
            try:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                    outputs = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **gen_kwargs
                    )
            except Exception as e_generate:
                raise RuntimeError(f"Error during model.generate: {e_generate}") from e_generate
        logger.debug("Generation completed.")
        if len(outputs) == 0:
            raise ValueError("No output generated by the model. Check the input and generation parameters.")
        output = outputs[0]  # Get the first generated sequence
        generated_tokens = (output[input_length:] if output.shape[0] > input_length else torch.tensor([], dtype=torch.long, device=device))
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
