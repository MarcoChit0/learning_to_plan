import os
from string import Template
from learning_to_plan.prompt_builder.pddl import pddl
from learning_to_plan import config
from learning_to_plan.data import task


LANDMARKS_EXAMPLE_TEMPLATE = Template(f"""{config.TOKENS.EXAMPLE_START.value}
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
    # TODO: ADD LANDMARK FACTORY TO PARSER
    def __init__(self, landmark_factory = "lm_zg", **kwargs):
        super().__init__(prompt_type=config.PROMPT_TYPE.LANDMARKS, **kwargs)
        self.landmark_factory = landmark_factory
        self.metadata["landmark_factory"] = landmark_factory

    def get_additional_info(self, t: task.Task) -> str:
        try:
            landmark_graph_str = get_landmark_graph(t, memory_limit=2, landmark_factory=self.landmark_factory)
        except Exception as e:
            raise ValueError(f"Error getting landmark graph for task {t.id}: {e}")
        try:
            landmarks = extract_landmarks(landmark_graph_str)
        except Exception as e:
            raise ValueError(f"Error extracting landmarks for task {t.id}: {e}")

        s = "For any planning problem, landmarks are the facts that must be true in all valid plans.\n\n"
        s += "The landmarks for this problem are as follows:\n\n"
        s += config.TOKENS.LANDMARKS_START.value + "\n"
        s += "\n".join(landmarks)
        s += "\n" + config.TOKENS.LANDMARKS_END.value + "\n"

        s += "\nHere are some examples of plans in the same format as the one you should provide:"

        for ex in self.examples:
            try:
                landmark_graph_str = get_landmark_graph(ex, memory_limit=24, landmark_factory=self.landmark_factory)
            except Exception as e:
                raise ValueError(f"Error getting landmark graph for task {ex.id}: {e}")
            try:
                landmarks = extract_landmarks(landmark_graph_str)
            except Exception as e:
                raise ValueError(f"Error extracting landmarks for task {ex.id}: {e}")
            
            s += "\n\n" + LANDMARKS_EXAMPLE_TEMPLATE.substitute(
                domain=ex.read_domain().strip(),
                instance=ex.read_instance().strip(),
                landmarks="\n".join(landmarks),
                plan=ex.get_plan()
            )

        return s


import subprocess
import resource

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
    
    return matches