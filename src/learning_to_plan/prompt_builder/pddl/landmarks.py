import os
from string import Template
from learning_to_plan.prompt_builder.pddl import pddl
from learning_to_plan import config
from learning_to_plan.data import task


BASIC_LANDMARKS_EXAMPLE_TEMPLATE = Template(f"""{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.LANDMARKS_START.value}
$landmarks
{config.TOKENS.LANDMARKS_END.value}
{config.TOKENS.PLAN_START.value}
$plan
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}
""")

THINKING_LANDMARKS_EXAMPLE_TEMPLATE = Template(f"""{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.LANDMARKS_START.value}
$landmarks
{config.TOKENS.LANDMARKS_END.value}
{config.TOKENS.THINKING_START.value}
$thinking
{config.TOKENS.THINKING_END.value}
{config.TOKENS.PLAN_START.value}
$plan
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}
""")

BASIC_LANDMARKS_PROMPT_TEMPLATE = Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. Landmarks are the facts that must be true in all valid plans. The landmarks section lists the landmarks for this problem. You must use the landmarks to help you generate the plan. Your response should only contain the plan.
                                
{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.LANDMARKS_START.value}
$landmarks
{config.TOKENS.LANDMARKS_END.value}

Here are some examples of plans in the same format as the one you should provide:

$examples

Here is a checklist to help you with your task:

{config.TOKENS.CHECKLIST_START.value}
1) Do not provide any additional text or explanations outside the plan tags.
2) The plan must be in the same format as the examples above.
3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
{config.TOKENS.CHECKLIST_END.value}""")

THINKING_LANDMARKS_PROMPT_TEMPLATE = Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. Landmarks are the facts that must be true in all valid plans. The landmarks section lists the landmarks for this problem. You must use the landmarks to help you generate the plan. You must provide the plan in your response. You may also provide your reasoning in the thinking section before the plan, which should be enclosed between {config.TOKENS.THINKING_START.value} and {config.TOKENS.THINKING_END.value} tags. Your response should only contain the reasoning and the plan.

{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.LANDMARKS_START.value}
$landmarks
{config.TOKENS.LANDMARKS_END.value}

Here are some examples of plans in the same format as the one you should provide:

$examples

Here is a checklist to help you with your task:

{config.TOKENS.CHECKLIST_START.value}
1) Your response could contain the reasoning inside the thinking section and must contain the plan inside the plan section. Do not provide any additional text or explanations outside these sections.
2) The plan must be in the same format as the examples above.
3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
6) You may provide your reasoning in the thinking section before the plan, which should be enclosed between {config.TOKENS.THINKING_START.value} and {config.TOKENS.THINKING_END.value} tags.
7) Your reasoning should be restricted to the thinking section. So, do not provide any reasoning inside the plan section.
{config.TOKENS.CHECKLIST_END.value}""")


"""
digraph G {
  lm0 [label="Atom on(a, d)", style=filled];
  lm1 [label="Atom clear(d)"];
      lm1 -> lm18 [label="n"];
      lm1 -> lm12 [label="n"];
      lm1 -> lm0 [label="n"];
  lm2 [label="Atom clear(a)"];
      lm2 -> lm18 [label="n"];
      lm2 -> lm16 [label="n"];
      lm2 -> lm12 [label="n"];
      lm2 -> lm0 [label="n"];
  lm3 [label="Atom clear(b)"];
      lm3 -> lm18 [label="n"];
      lm3 -> lm16 [label="n"];
      lm3 -> lm12 [label="n"];
      lm3 -> lm0 [label="n"];
  lm4 [label="Atom clear(c)", style=bold];
      lm4 -> lm18 [label="n"];
      lm4 -> lm16 [label="n"];
      lm4 -> lm12 [label="n"];
      lm4 -> lm0 [label="n"];

  lm13 [label="Atom on(d, e)", style=bold];
      lm13 -> lm18 [label="n"];
      lm13 -> lm12 [label="n"];
  lm14 [label="Atom holding(b)"];
      lm14 -> lm12 [label="n"];
  lm15 [label="Atom clear(e)"];
      lm15 -> lm18 [label="n"];
      lm15 -> lm12 [label="n"];
  lm16 [label="Atom on(c, f)", style=filled];
  lm17 [label="Atom holding(c)"];
      lm17 -> lm16 [label="n"];
  lm18 [label="Atom on(e, c)", style=filled];
  lm19 [label="Atom ontable(e)", style=bold];
      lm19 -> lm18 [label="n"];
  lm20 [label="Atom holding(e)"];
      lm20 -> lm18 [label="n"];
}
"""

class LandmarksPromptBuilder(pddl.PDDLPromptBuilder):
    MEMORY_LIMIT = 2 # GB
    def __init__(self, landmark_factory = "lm_zg", **kwargs):
        self.landmark_factory = landmark_factory
        super().__init__(prompt_type=config.PROMPT_TYPE.LANDMARKS, **kwargs)
        self.metadata["landmark_factory"] = landmark_factory
        for example in self.examples:
            if "landmarks" not in self.examples_data[example]:
                try:
                    landmarks : list[str] = get_landmarks(example, self.MEMORY_LIMIT, self.landmark_factory)
                except Exception as e:
                    raise ValueError(f"Error getting landmarks for example {example.id}: {e}")
                self.examples_data[example]["landmarks"] = landmarks
            else:
                landmarks = self.examples_data[example]["landmarks"].splitlines()

            if "thinking" in self.examples_data[example]:
                self.examples_data[example]["thinking"].add_flag_to_state_facts(
                    "landmark",
                    landmarks
                )

    def get_example_template(self, **kwargs) -> Template:
        """
        Returns the template to be used for generating the prompt.
        :param kwargs: Additional keyword arguments.
        :return: The template to be used for generating the prompt.
        """
        if self.thinking_style == config.THINKING_STYLE.NONE:
            return BASIC_LANDMARKS_EXAMPLE_TEMPLATE
        elif self.thinking_style == config.THINKING_STYLE.COT:
            return THINKING_LANDMARKS_EXAMPLE_TEMPLATE
        else:
            raise ValueError(f"Unknown thinking style: {self.thinking_style}. Please provide a valid thinking style.")
    
    def get_prompt_template(self, t : task.Task, **kwargs) -> Template:
        """
        Returns the prompt template to be used for generating the prompt.
        :param kwargs: Additional keyword arguments.
        :return: The prompt template to be used for generating the prompt.
        """
        if self.thinking_style == config.THINKING_STYLE.NONE:
            return BASIC_LANDMARKS_PROMPT_TEMPLATE
        elif self.thinking_style == config.THINKING_STYLE.COT:
            return THINKING_LANDMARKS_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Unknown thinking style: {self.thinking_style}. Please provide a valid thinking style.")
    
    def get_task_data(self, t, with_plan):
        _data = super().get_task_data(t, with_plan)
        try:
            landmarks = get_landmarks(t, self.MEMORY_LIMIT, self.landmark_factory)
        except Exception as e:
            raise ValueError(f"Error getting landmarks for task {t.id}: {e}")
        _data["landmarks"] = "\n".join(landmarks)
        return _data

import subprocess
import resource

def get_landmarks(t : task.Task, memory_limit: int = 24, landmark_factory: str = "lm_zg") -> list[str]:
    try:
        landmark_graph_str = get_landmark_graph(t, memory_limit=memory_limit, landmark_factory=landmark_factory)
    except Exception as e:
        raise ValueError(f"Error getting landmark graph for task {t.id}: {e}")
    try:
        return extract_landmarks(landmark_graph_str)
    except Exception as e:
        raise ValueError(f"Error extracting landmarks for task {t.id}: {e}")

def get_landmark_graph(t : task.Task, memory_limit: int = 24, landmark_factory:str = "lm_zg") -> None:
    # command = ./utils/downward/fast-downward.py ./data/raw/blocksworld/generated_domain.pddl ./data/raw/blocksworld/generated_basic_longer_plan_len/instance-20.pddl --search "lazy_greedy([landmark_sum(lm_zg(verbosity=debug))])"
    cmd_list = [
        './utils/downward/fast-downward.py',
        t.domain_file_path,
        t.instance_file_path,
        '--search',
        f"lazy_greedy([landmark_sum({landmark_factory}(verbosity=debug))])"
    ]
        
    
    def set_memory_limit():
        # Set memory limit (adjust as needed)
        mem_in_gbyte = memory_limit * 1024 * 1024 * 1024
        # Get current limits to avoid raising above hard limit
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        # Only set if the requested limit is lower than current hard limit
        if current_hard == resource.RLIM_INFINITY or mem_in_gbyte <= current_hard:
            resource.setrlimit(resource.RLIMIT_AS, (mem_in_gbyte, current_hard))
    
    try:
        result = subprocess.run(
            cmd_list, 
            capture_output=True, 
            text=True, 
            check=True,
            preexec_fn=set_memory_limit  # Pass function reference, not call
        )
        if os.path.exists('./sas_plan'):
            os.remove('./sas_plan')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error calling Downward: {e.stderr.strip()}")
    reading_graph = False
    graph_lines = []
    for line in result.stdout.splitlines():
        if reading_graph:
            if line.startswith("[t="):
                reading_graph = False
                continue
            graph_lines.append(line)
        if "Dumping landmark graph:" in line:
            reading_graph = True
            continue
    
    if not graph_lines:
        raise ValueError("No landmark graph found in the output of Downward.")
    graph = "\n".join(graph_lines)
    # check on the start of the graph "digraph G {"
    if not graph.startswith("digraph G {"):
        raise ValueError("The landmark graph does not start with 'digraph G {'.")
    # check on the end of the graph "}"
    if not graph.endswith("}"):
        raise ValueError("The landmark graph does not end with '}'.")
    return graph


def extract_landmarks(landmark_graph_str: str) -> list[str]:
    import re
    # Use regex to extract labels from the dot graph instead of pydot to avoid memory issues
    label_pattern = r'label="Atom ([^"]+)"'
    matches = re.findall(label_pattern, landmark_graph_str)
    
    if not matches:
        raise ValueError(f"No landmarks found in the landmark graph.")
    
    # Each predicate is on the following format <name>(<operand1>, <operand2>, ...)
    # It is need to convert to the format (<name> <operand1> <operand2> ...)
    predicate_pattern = r'(\w+)\((.*)\)'
    operand_pattern = r'\s*(\w+)\s*,?\s*'
    landmarks = []
    for match in matches:
        # Extract the predicate name and operands
        predicate_match = re.match(predicate_pattern, match)
        if not predicate_match:
            raise ValueError(f"Invalid landmark format: {match}")
        
        predicate_name = predicate_match.group(1)
        operands = re.findall(operand_pattern, predicate_match.group(2))

        # Format the landmark as "(<name> <operand1> <operand2> ...)"
        formatted_landmark = f"({predicate_name} {' '.join(operands).strip()})" if len(operands) > 0 else f"({predicate_name})"
        landmarks.append(formatted_landmark)

    return landmarks