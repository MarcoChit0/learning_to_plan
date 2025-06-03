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
from learning_to_plan import config, task
import torch
import datasets
import numpy as np
import json
logger = config.get_logger(__name__)

class Model:
    VALID_PROMPT_TYPES = ["io", "cot"]
    def __init__(self, model_name, **kwargs):
        """
        task:task.Task : { # Task object for which the plan was generated
            prompt_type:str : { # Prompt type used for generation, e.g., "io", "cot", ...
                "raw" : list[str], # List of raw generated plans
                "pddl" : list[str] # List of PDDL generated plans
                "is_valid" : list[Optional[bool]] # List of booleans indicating if the plan is valid. If None, the plan is not validated.
            }
        }
        """
        self._model_name = model_name
        self.__dict__.update(kwargs)
        self._generated_plans:dict[task.Task, dict[str, Any]] = {}
        self._model_dir_path = os.path.join(config.MODELS_DIR, model_name)
        if kwargs.get("reset_model_dir", False):
            if os.path.exists(self._model_dir_path):
                logger.info(f"Deleting existing model directory: {self._model_dir_path}")
                os.rmdir(self._model_dir_path)
        os.makedirs(self._model_dir_path, exist_ok=True)

    def add_generated_plans(self, task: task.Task, prompt_type: str, raw_plans:list[str], pddl_plans:list[str], is_valid:Optional[list[Optional[bool]]] = None, overwrite:bool=False) -> None:
        """
            Adds generated plans to the internal dictionary.
            This method is called after generating plans for a task.

            Parameters:
                task: The task object for which the plans were generated.
                prompt_type: The type of prompt used for generation (e.g., "io", "cot").
                outputs: The generated outputs from the model.
                input_length: The length of the input tokens.
                device: The device on which the model is running (e.g., "cuda", "cpu").
        """

        if  len(raw_plans) != len(pddl_plans):
            raise ValueError(f"Length mismatch: raw_plans ({len(raw_plans)}), pddl_plans ({len(pddl_plans)})")
        
        if is_valid is not None and len(raw_plans) != len(is_valid):
            raise ValueError(f"Length mismatch: raw_plans ({len(raw_plans)}), is_valid ({len(is_valid)})")

        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type: {prompt_type}. Must be in [{', '.join(self.VALID_PROMPT_TYPES)}].")
        
        if task not in self._generated_plans:
            self._generated_plans[task] = {}
        
        if prompt_type not in self._generated_plans[task] or overwrite:
            self._generated_plans[task][prompt_type] = {
                "raw": [],
                "pddl": [],
                "is_valid": [],
            }
        
        if overwrite:
            logger.info(f"Overwriting existing plans for task {task} and prompt type {prompt_type}.")
        
        self._generated_plans[task][prompt_type]["raw"].extend(raw_plans)
        self._generated_plans[task][prompt_type]["pddl"].extend(pddl_plans)
        logger.info(f"Added {len(raw_plans)} plans for task {task} and prompt type {prompt_type}.")
        if is_valid is not None:
            logger.info(f"The {len(is_valid)} plans added were validated.")
            self._generated_plans[task][prompt_type]["is_valid"].extend(is_valid)
        else:
            logger.info(f"The {len(raw_plans)} plans added were not validated.")
            self._generated_plans[task][prompt_type]["is_valid"].extend([None] * len(raw_plans))
    
    def validate_generated_plan(self, task:task.Task, prompt_type:str, plan_idx:int, is_valid:bool) -> None:
        """
        Validates a generated plan for a specific task and prompt type.
        This method is called after generating plans for a task.
        """
        if task not in self._generated_plans:
            raise ValueError(f"Task {task} not found in generated plans.")
        
        if prompt_type not in self._generated_plans[task]:
            raise ValueError(f"Prompt type {prompt_type} not found in generated plans for task {task}.")
        
        if plan_idx < 0 or plan_idx >= len(self._generated_plans[task][prompt_type]["raw"]):
            raise IndexError(f"Plan index {plan_idx} out of range for task {task} and prompt type {prompt_type}.")
        
        self._generated_plans[task][prompt_type]["is_valid"][plan_idx] = is_valid

        if is_valid:
            s = "is valid."
        else:
            s = "is invalid."
        logger.debug(f"Model {self._model_name} - Task {task} - Prompt Type {prompt_type} - Plan Index {plan_idx}: Plan validated as {s}.")

    def generate(self, task:task.Task, prompt_type:task.Task.PromptType,  **generation_kwargs) -> None:
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

    def tokenize_chat(self, examples, max_seq_length: int = 1024) -> Dict[str, Any]:
        """
        Tokenizes the input examples for chat-based models.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def decode(self, input_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes the input IDs to a string.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def save_generated_plans(self) -> None:
        """
        Saves the generated plans to a JSONL file.
        Each line in the file corresponds to a (task, prompt) pair.
        In each line, there is a JSON object containing:
        - task domain file path
        - task instance file path
        - prompt type
        - raw plans
        - pddl plans
        """
        file_path = os.path.join(self._model_dir_path, config.GENERATED_PLANS_FILE_NAME)
        logger.info(f"Saving generated plans to {file_path}.")
        number_of_tasks = self._generated_plans.keys()
        number_of_lines = 0
        number_of_plans = 0
        with open(file_path, "w", encoding="utf-8") as f:
            for task_obj, prompts in self._generated_plans.items():
                for prompt_type, plan_data in prompts.items():
                    output = {
                        "domain_file_path": getattr(task_obj, "_domain_file_path", None),
                        "instance_file_path": getattr(task_obj, "_instance_file_path", None),
                        "prompt_type": prompt_type,
                        "raw_plans": plan_data.get("raw", []),
                        "pddl_plans": plan_data.get("pddl", []),
                        "is_valid": plan_data.get("is_valid", []),
                    }
                    f.write(json.dumps(output) + "\n")
                    number_of_lines += 1
                    number_of_plans += len(plan_data.get("raw", []))
        logger.info(f"Saved {number_of_lines} lines to {file_path}.")
        logger.info(f"Saved {number_of_plans} plans from {len(number_of_tasks)} tasks.")
    
    def load_generated_plans(self) -> None:
        """
        Loads generated plans from a JSONL file and overwrites the current generated_plans.
        Assumes each line in the file is a JSON object with:
        - domain_file_path
        - instance_file_path
        - prompt_type
        - raw_plans
        - pddl_plans
        A pseudo task key is created as a tuple (domain_file_path, instance_file_path).
        """
        loaded_plans = {}
        file_path = os.path.join(self._model_dir_path, config.GENERATED_PLANS_FILE_NAME)
        logger.info(f"Loading generated plans from {file_path}.")
        number_of_tasks = 0
        number_of_plans = 0
        number_of_lines = 0
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist. No plans loaded.")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                number_of_lines += 1
                if line.strip():
                    data = json.loads(line)
                    t = task.get_task(
                        domain_file_path=data.get("domain_file_path"),
                        instance_file_path=data.get("instance_file_path"),
                    )
                    if t not in loaded_plans:
                        loaded_plans[t] = {}
                        number_of_tasks += 1
                    
                    loaded_plans[t][data.get("prompt_type")] = {
                        "raw": data.get("raw_plans", []),
                        "pddl": data.get("pddl_plans", []),
                        "is_valid": data.get("is_valid", []),
                    }
                    number_of_plans += len(data.get("raw_plans", []))
        self._generated_plans = loaded_plans
        logger.info(f"{number_of_lines} lines read from {file_path}.")
        logger.info(f"Loaded {number_of_plans} plans from {number_of_tasks} tasks.")
                
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
            all_config_tokens = [getattr(config.TOKENS, attr) for attr in dir(config.TOKENS) if not callable(getattr(config.TOKENS, attr)) and not attr.startswith("__")]

            for token_str in all_config_tokens:
                if isinstance(token_str, str) and token_str not in self._tokenizer.get_vocab():
                    special_tokens_to_add.append(token_str)

            if special_tokens_to_add:
                logger.info(f"Added {len(special_tokens_to_add)} special tokens to tokenizer: {special_tokens_to_add}")

            logger.info(f"Tokenizer loaded. Pad token: {self._tokenizer.pad_token}, Padding side: {self._tokenizer.padding_side}")

            for token in special_tokens_to_add:
                if token not in self._tokenizer.get_vocab():
                    raise ValueError(f"Special token '{token}' not found in tokenizer vocabulary. It will be added.")
                else:
                    logger.info(f"Special token '{token}' added to tokenizer vocabulary.")

        except Exception as e:
            logger.error(f"Error loading or setting up tokenizer from {model_source}: {e}", exc_info=True)
            raise e

        # --- Model Loading ---
        try:
            torch_dtype = torch.bfloat16 if self.__dict__.get("bf16", False) else torch.float16
            
            quantization_config_param = None
            if self.__dict__.get("load_in_8bit", False): # Check if load_in_8bit is true from train_config.json
                quantization_config_param = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("8-bit quantization enabled for model loading.")

            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=config.HUGGINGFACE_TOKEN,
                device_map="auto",
                quantization_config=quantization_config_param,
            )
            logger.info(f"Model loaded successfully from {model_source}.")

            # --- Resize embeddings if tokens were added ---
            if self._tokenizer.vocab_size != self._model.get_input_embeddings().weight.shape[0]:
                logger.info(f"Resizing model token embeddings to match tokenizer size: {len(self._tokenizer)}")
                self._model.resize_token_embeddings(len(self._tokenizer))
                logger.info("Model token embeddings resized successfully.")
                # ---- Initialize new token embeddings with almost zero values ----
                if not last_checkpoint and special_tokens_to_add:
                    embedding_layer = self._model.get_input_embeddings()
                    reference_token = "<|endoftext|>"
                    reference_token_id = self._tokenizer.convert_tokens_to_ids(reference_token)
                    logger.info(
                        msg=f"Token {', '.join(special_tokens_to_add)} cannot initialize its embedding layer with almost zero values. "
                        f"Setting it to the same as the reference token ID: {reference_token}, {reference_token_id}."
                    )
                    with torch.no_grad():
                        for token in special_tokens_to_add:
                            token_id = self._tokenizer.convert_tokens_to_ids(token)
                            embedding_layer.weight[token_id].copy_(embedding_layer.weight[reference_token_id])

            if quantization_config_param and kwargs.get("is_trainable", False):
                logger.info("Preparing model for 8-bit training.")
                self._model = prepare_model_for_kbit_training(self._model, use_gradient_checkpointing=True)
                logger.info("Model prepared for 8-bit training.")
            
            if last_checkpoint:
                logger.info(f"Loading model state from checkpoint: {last_checkpoint}")
                self._model = PeftModel.from_pretrained(
                    self._model,
                    last_checkpoint,
                    token=config.HUGGINGFACE_TOKEN,
                    is_trainable=kwargs.get("is_trainable", False),
                )
                logger.info(f"Model state loaded from checkpoint: {last_checkpoint}")

        except Exception as e:
            logger.error(f"Error loading model from {model_source} or resizing embeddings: {e}", exc_info=True)
            raise e

    def get_token_probability(self, input_tokens, target_token):
        with torch.no_grad():
            outputs = self._model(input_tokens)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        token_prob = probs[0, target_token]
        return token_prob


    def get_token_probability(self, input_tokens, target_token):
        with torch.no_grad():
            outputs = self._model(input_tokens)
        # get the logits for our model output
        logits = outputs.logits[:, -1, :]
        # calculate the softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        token_prob = probs[0, target_token]
        return token_prob


    def decode(self, input_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(
            input_ids,
            skip_special_tokens=skip_special_tokens
        )

    def tokenize_chat(self, chat:list[dict[str, str]], max_seq_length: int = 1024) -> Dict[str, Any]:
        """
        Tokenizes a single chat conversation for training.
        """
        # --- Tokenize the chat conversation ---
        tokenized_chat = self._tokenizer.apply_chat_template(
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
            if labels[i] == self._tokenizer.pad_token_id:
                labels[i] = -100



        # --- verify the plan ---
        # The plan is between the start and end tokens (inclusive)
        PLAN_START_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.TOKENS.PLAN_START)
        PLAN_END_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.TOKENS.PLAN_END)
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
        if not isinstance(self._model, PeftModel):
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
            warmup_ratio=train_kwargs.get("warmup_ratio", 0.1),
            label_names=["labels"],
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
            self._model.save_pretrained(checkpoint_dir, save_embedding_layers=True) # Recommended for PEFT models
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

    def generate(self, task:task.Task, prompt_type:task.Task.PromptType, **generation_kwargs) -> None:
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

        num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
        overwrite_plans = generation_kwargs.get("overwrite_generated_plans", False)
        if task in self._generated_plans and prompt_type in self._generated_plans[task]:
            if not overwrite_plans:
                num_already_generated_plans = len(self._generated_plans[task][prompt_type]['raw'])
                logger.info(f"Task {task._id} already has {num_already_generated_plans} generated plans with prompt type {prompt_type}.")
                if num_already_generated_plans >= num_return_sequences:
                    logger.info(f"Skipping generation for task {task._id} with prompt type {prompt_type}.")
                    return
                else:
                    logger.info(f"Task {task._id} has {num_already_generated_plans} generated plans with prompt type {prompt_type}. Generating {num_return_sequences - num_already_generated_plans} more plans to match the requested {num_return_sequences} plans.")
                    num_return_sequences -= num_already_generated_plans
            else:
                logger.info(f"Overwriting existing plans for task {task._id} with prompt type {prompt_type}.")

        # Ensure model is in evaluation mode
        self._model.eval()

        device = next(self._model.parameters()).device

        generation_messages: list[dict[str, str]] = task.get_chat(with_plan=False, prompt_type=prompt_type, **generation_kwargs) 
        
        # --- Generation Configuration ---
        gen_kwargs = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 512),
            "do_sample": generation_kwargs.get("do_sample", True),
            "temperature": generation_kwargs.get("temperature", 0.7),
            "top_p": generation_kwargs.get("top_p", 0.93),
            "top_k": generation_kwargs.get("top_k", 50),
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
            "num_return_sequences": num_return_sequences,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        logger.debug(f"Generation parameters: {gen_kwargs}")
        try:
            # Set add_generation_prompt=False so the model continues from the provided assistant prompt
            inputs = self._tokenizer.apply_chat_template(
                generation_messages,
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
            dtype = getattr(self._model, 'dtype', torch.float16)
            use_autocast = (device.type == 'cuda') and dtype in [torch.float16, torch.bfloat16]
            
            try:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                    outputs = self._model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **gen_kwargs
                    )
            except Exception as e_generate:
                raise RuntimeError(f"Error during model.generate: {e_generate}") from e_generate
        
        # --- Process Outputs ---
        raw_outputs = []
        pddl_plans = []
        for output in outputs:
            generated_tokens = (output[input_length:] if output.shape[0] > input_length else torch.tensor([], dtype=torch.long, device=device))
            generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=False)
            raw_outputs.append(generated_text)

            # # --- Process Plan ---
            # TODO: use this in the future when the model knows how to add the plan start and end tokens
            # START_OF_PLAN_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.START_OF_PLAN_TOKEN)
            # END_OF_PLAN_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.END_OF_PLAN_TOKEN)
            
            # start_of_plan_idx = next((i for i, token in enumerate(output) if token == START_OF_PLAN_TOKEN_ID), None)
            # if start_of_plan_idx:
            #     end_of_plan_idx = next((i for i, token in enumerate(output[start_of_plan_idx:]) if token == END_OF_PLAN_TOKEN_ID), None)
            #     if end_of_plan_idx:
            #         plan_tokens = output[start_of_plan_idx:end_of_plan_idx + 1]
            #         plan_text = self._tokenizer.decode(plan_tokens, skip_special_tokens=True)
            #         processed_outputs.append(plan_text)
            #         logger.info(f"Generated plan for task {task._id} with prompt type {prompt_type}: {plan_text}")
            #     else:
            #         processed_outputs.append("Generation Error: No end of plan token found.")
            #         logger.info(f"Generated plan for task {task._id} with prompt type {prompt_type}: No end of plan token found in output tokens [{output[:100]}...]")
            # else:
            #     processed_outputs.append("Generation Error: No start of plan token found.")
            #     logger.info(f"Generated plan for task {task._id} with prompt type {prompt_type}: No start of plan token found in output tokens [{output[:100]}...]")

            # TODO: remove this when the model knows how to add the plan start and end tokens
            pddl_plans.append(task._domain_translator.translate_natural_language_plan_to_pddl(generated_text))
            print(f"Generated PDDL plan:\n{pddl_plans[-1]}")
        self.add_generated_plans(
            task=task, 
            prompt_type=prompt_type, 
            raw_plans=raw_outputs,
            pddl_plans=pddl_plans, 
            overwrite=overwrite_plans
        )

        logger.info(f"Generated {len(raw_outputs)} plans for task {task} with prompt type {prompt_type}.")


# # --- Gemini Model (Remains unchanged from previous version) ---
import google.generativeai as genai
class GeminiModel(Model):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        assert config.GOOGLE_API_KEY, "Google API Key is required for Gemini model."
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            logger.info("Gemini API configured successfully.")
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
            task:task.Task,
            prompt_type:task.Task.PromptType,
            **generation_kwargs:dict[str, Any]
        ) -> None:
        logger.debug(f"Generating with Gemini model {self._model_name}.")

        original_chat_messages: list[dict[str, str]] = task.get_chat(with_plan=False, prompt_type=prompt_type, **generation_kwargs)

        system_instruction_parts = []
        gemini_contents = []

        for msg in original_chat_messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")

            if role == "system":
                system_instruction_parts.append(content)
            elif role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant": # Map "assistant" to "model" for Gemini
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            else:
                logger.warning(f"Unsupported role '{role}' in chat for Gemini model. Skipping message: '{content[:50]}...'")
        
        final_system_instruction = "\n".join(system_instruction_parts) if system_instruction_parts else None
        
        generation_config = genai.types.GenerationConfig( # Use GenerationConfig object
            temperature=generation_kwargs.get("temperature", 0.7),
            top_p=generation_kwargs.get("top_p", 0.93),
            top_k=generation_kwargs.get("top_k", 50),
            max_output_tokens=generation_kwargs.get("max_output_tokens", 2048),
            candidate_count=generation_kwargs.get("candidate_count", 1), # Map num_return_sequences
            stop_sequences=[config.TOKENS.PLAN_END] # Add plan end token as stop sequence
        )
        logger.debug(f"Gemini generation config: {generation_config}")


        try:
            model = genai.GenerativeModel(
                self._model_name,
                generation_config=generation_config,
                system_instruction=final_system_instruction,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{self._model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Gemini model '{self._model_name}': {e}") from e

        try:
            wait_time = generation_kwargs.get("wait_time", 0) # Default to 0 wait time unless specified
            if wait_time > 0:
                logger.info(f"Waiting for {wait_time} seconds before Gemini API call.")
                time.sleep(wait_time)

            logger.debug("Calling Gemini model.generate_content...")
            # The prompt should ideally include PLAN_START_TOKEN if Gemini needs it to trigger plan generation
            # Example: prompt_text_full = prompt_text + PLAN_START_TOKEN
            response = model.generate_content(gemini_contents) # Use original prompt_text for now
            logger.debug("Gemini API call completed.")


            generated_texts = []
            if response and response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        generated_texts.append(text.strip())
                    elif candidate.content and not candidate.content.parts:
                         logger.warning(f"Gemini candidate content has no parts: {candidate.content}")
                    else:
                         logger.warning(f"Gemini candidate has no content: {candidate}")

            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 logger.error(f"Gemini request blocked. Reason: {response.prompt_feedback.block_reason}")
                 raise RuntimeError(f"Gemini request blocked. Reason: {response.prompt_feedback.block_reason}")
            if not generated_texts and response.candidates:
                 finish_reasons = [c.finish_reason for c in response.candidates]
                 logger.warning(f"No text extracted from Gemini response. Finish reasons: {finish_reasons}")


            if not generated_texts:
                logger.error(f"No valid generated texts found in Gemini response. Response: {response}")
                raise RuntimeError("No valid generated texts found in Gemini response.")

            pddl_plans = [task._domain_translator.translate_natural_language_plan_to_pddl(text) for text in generated_texts]
            prompt_type = "cot" if len(cot_examples) > 0 else "io"
            self.add_generated_plans(
                task=task,
                prompt_type=prompt_type,
                raw_plans=generated_texts,
                pddl_plans=pddl_plans,
                overwrite=generation_kwargs.get("overwrite_generated_plans", False)
            )
            logger.info(f"Generated {len(generated_texts)} plans for task {task} with prompt type {prompt_type}.")

        except Exception as e:
            logger.error(f"Failed to generate text with Gemini model '{self._model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate text with Gemini model '{self._model_name}': {e}") from e

# --- get_model function (Remains unchanged) ---
def get_model(model_name: str, **kwargs) -> Model:
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
        model = model_cls(model_name, **kwargs)
    except Exception as e:
        logger.error(f"Error creating model instance: {e}", exc_info=True)
        raise e
    model.load_generated_plans()
    return model

