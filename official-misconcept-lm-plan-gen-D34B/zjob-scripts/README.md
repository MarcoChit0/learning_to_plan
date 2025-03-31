## Scripts for the Experiments
Our experiments are built using the [Kedro Framework](https://kedro.org/), which we have carefully chosen to ensure the reproducibility of our experiments. The scripts are located in the `zjob-scripts` directory. The following is a brief description of the scripts:

`llama_fac_config/`: Contains the configuration files for LLM planner training. We used Llama factory framework to train the LLM planner.

`step_1_huggingface_dataset_creation.sh`: This script is used to create the dataset for the experiments. Normally you do not need to run this script, as the dataset will be available on the Hugging Face Hub once the paper reviewing process is complete.


`step_2_training_llm_for_plan_gen.sh`: Training the LLM planner for plan generation. This script trains the LLM planner using the dataset created in the previous step.

`step_3_llama_fac_lm_planning_evaluation.sh`: Evaluating the LLM planner. This script evaluates the LLM planner on the test set. Make sure that you update the configuration file located in the `conf/base/parameters_language_model_planning_evaluation.yml` before running this script.

`step_3_llama_fac_lm_planning_evaluation_ppo.sh`: A variant of the evaluation script that evaluate the model that is further trained by RL, note that we do not actually use PPO but rather RLOO (see the paper for details).


`step_4_lm_planning_probing.sh`: This script runs the mistake recognition probing test on the LLM planner. Make sure that you update the configuration file located in the `conf/base/parameters_plan_validation_probing.yml` before running this script.

> [!WARNING]
> In fact, you have to make sure you updated the configuration files in the `conf/base` directory before running the scripts. If you encounter issues when running the scripts, most likely the configuration files are not correctly set up.