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
                "processed" : list[str] # List of processed generated plans
                "pddl" : list[str] # List of PDDL generated plans
                "is_valid" : list[Optional[bool]] # List of booleans indicating if the plan is valid. If None, the plan is not validated.
            }
        }
        """
        self._model_name = model_name
        self.__dict__.update(kwargs)
        self._generated_plans:dict[task.Task, dict[str, Any]] = {}
        self._model_dir_path = os.path.join(config.MODELS_DIR, model_name)
        # TODO: add this to args
        if kwargs.get("reset_model_dir", False):
            if os.path.exists(self._model_dir_path):
                logger.info(f"Deleting existing model directory: {self._model_dir_path}")
                os.rmdir(self._model_dir_path)
        os.makedirs(self._model_dir_path, exist_ok=True)

    def add_generated_plans(self, task: task.Task, prompt_type: str, raw_plans:list[str], processed_plans:list[str], pddl_plans:list[str], is_valid:Optional[list[Optional[bool]]] = None, overwrite:bool=False) -> None:
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

        if len(raw_plans) != len(processed_plans) or len(raw_plans) != len(pddl_plans):
            raise ValueError(f"Length mismatch: raw_plans ({len(raw_plans)}), processed_plans ({len(processed_plans)}), pddl_plans ({len(pddl_plans)})")
        
        if is_valid is not None and len(raw_plans) != len(is_valid):
            raise ValueError(f"Length mismatch: raw_plans ({len(raw_plans)}), is_valid ({len(is_valid)})")

        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type: {prompt_type}. Must be in [{', '.join(self.VALID_PROMPT_TYPES)}].")
        
        if task not in self._generated_plans:
            self._generated_plans[task] = {}
        
        if prompt_type not in self._generated_plans[task] or overwrite:
            self._generated_plans[task][prompt_type] = {
                "raw": [],
                "processed": [],
                "pddl": [],
                "is_valid": [],
            }
        
        if overwrite:
            logger.info(f"Overwriting existing plans for task {task} and prompt type {prompt_type}.")
        
        self._generated_plans[task][prompt_type]["raw"].extend(raw_plans)
        self._generated_plans[task][prompt_type]["processed"].extend(processed_plans)
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
        logger.info(f"Model {self._model_name} - Task {task} - Prompt Type {prompt_type} - Plan Index {plan_idx}: Plan validated as {s}.")

    def generate(self, task:task.Task, cot_examples:set[task.Task]=set(), **generation_kwargs) -> None:
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
        - processed plans
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
                        "processed_plans": plan_data.get("processed", []),
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
        - processed_plans
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
                        "processed": data.get("processed_plans", []),
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
                token=config.HUGGINGFACE_TOKEN
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
                    is_trainable=kwargs.get("is_trainable", False),
                )
                logger.info(f"Model state loaded from checkpoint: {last_checkpoint}")
            else:
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

    def tokenize_chat(self, examples: Dict[str, List[str]], max_seq_length: int = 1024) -> Dict[str, Any]:
        """
        Tokenizes a batch of examples using the chat template.
        'examples' is a dictionary where keys like "instruction", "input", "output"
        hold lists of strings (one string per example in the batch).
        """
        dataset_len = len(examples["instruction"])
        if not (dataset_len == len(examples["input"]) == len(examples["output"])):
            raise ValueError("Instruction, input, and output lists must have the same length.")
        
        if dataset_len == 0:
            logger.warning("tokenize_chat received an empty batch of examples.")
            return {"input_ids": [], "attention_mask": [], "labels": []}


        batch_messages: List[List[Dict[str, str]]] = []

        for i in range(dataset_len):
            user_content = examples["instruction"][i] + "\n" + examples["input"][i]
            assistant_content = examples["output"][i]

            full_conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            batch_messages.append(full_conversation)
        
        # Tokenize the full conversations for training
        # Force PyTorch tensor output to ensure BatchEncoding (dict-like) is returned
        tokenized_encoding_batch = self._tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=False,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_dict=True
        )

        # Convert tensors to lists for subsequent Python logic
        processed_tokenized_outputs = {
            "input_ids": tokenized_encoding_batch["input_ids"].tolist(),
            "attention_mask": tokenized_encoding_batch["attention_mask"].tolist(),
            "labels": []
        }

        labels_batch = []
        PLAN_START_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.START_OF_PLAN_TOKEN)
        PLAN_END_TOKEN_ID = self._tokenizer.convert_tokens_to_ids(config.END_OF_PLAN_TOKEN)
        for i in range(dataset_len): 
            # --- check plan start and end tokens ---
            plan_start_idx = next((
                idx for idx, token_id in enumerate(processed_tokenized_outputs["input_ids"][i]) if token_id == PLAN_START_TOKEN_ID), None)
            
            if plan_start_idx is None:
                raise ValueError(f"PLAN_START_TOKEN_ID {PLAN_START_TOKEN_ID} not found in input_ids for example {i}.")
            
            plan_end_idx = next((
                idx for idx, token_id in enumerate(processed_tokenized_outputs["input_ids"][i], start=plan_start_idx) if token_id == PLAN_END_TOKEN_ID), None)
            
            if plan_end_idx is None:
                raise ValueError(f"PLAN_END_TOKEN_ID {PLAN_END_TOKEN_ID} not found in input_ids for example {i} after plan_start_idx.")
            
            # --- labels ---
            labels = [-100] * len(processed_tokenized_outputs["input_ids"][i])
            labels[plan_start_idx:plan_end_idx + 1] = processed_tokenized_outputs["input_ids"][i][plan_start_idx:plan_end_idx + 1]

            labels_batch.append(labels)
        
        processed_tokenized_outputs["labels"] = labels_batch
        return processed_tokenized_outputs

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

    def generate(self, task: task.Task, cot_examples:set[task.Task]=set(), **generation_kwargs) -> None:
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
        overwrite_plans = generation_kwargs.get("overwrite_plans", False)
        prompt_type = "cot" if len(cot_examples) > 0 else "io"
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

        prompt_components = task.get_prompt_componenets()
        generation_messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_components["instruction"] + "\n" + prompt_components["input"]},
        ]
        
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
        processed_outputs = []
        raw_outputs = []
        pddl_plans = []
        for output in outputs:
            generated_tokens = (output[input_length:] if output.shape[0] > input_length else torch.tensor([], dtype=torch.long, device=device))
            generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
            processed_outputs.append(generated_text)
            pddl_plans.append(task._converter.natural_language_plan_to_pddl(generated_text))
            print(f"Generated PDDL plan:\n{pddl_plans[-1]}")
        self.add_generated_plans(
            task=task, 
            prompt_type=prompt_type, 
            raw_plans=raw_outputs, 
            processed_plans=processed_outputs, 
            pddl_plans=pddl_plans, 
            overwrite=overwrite_plans
        )

        logger.info(f"Generated {len(processed_outputs)} plans for task {task} with prompt type {prompt_type}.")


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
    model = HuggingFaceModel(model_name, **kwargs)
    model.load_generated_plans()
    return model

