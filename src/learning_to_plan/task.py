from __future__ import annotations
from copy import deepcopy
import threading
import abc
import re
import json
import os
from learning_to_plan import config
from typing import Optional
from enum import Enum

logger = config.get_logger(__name__)

instance_pattern = re.compile(r"instance-(\d+)\.pddl$")
lock = threading.Lock()

class Task(abc.ABC):
    class TYPE(Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    class PAAS_STATUS(Enum):
        OK = "ok"
        ERROR = "error"
    
    class DomainTranslator(abc.ABC):
        FACT_PATTERN = r"\(\s*[\w-]+\s*[\w\s-]*\)"

        def get_section_content(self, section_name: str, text: str) -> str | None:
            # --- Helper function to extract section content ---
            # Pattern explanation:
            # \(:section_name\s+   : Match the keyword (e.g., :init) followed by whitespace
            # (.*?)                : Non-greedily capture the content (re.DOTALL makes '.' match newline)
            # (?=\s+\(:|\s*\)\s*$) : Positive lookahead assertion:
            #    \s+\(:            : Ensure the match is followed by whitespace and the start of another section '(:...'
            #    |                 : OR
            #    \s*\)\s*$         : Ensure the match is followed by the final closing parenthesis of the define block
            #                       (allowing for potential whitespace)
            pattern = r"\({}\s+(.*?)(?=\s+\(:|\s*\)\s*$)".format(re.escape(section_name))
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() # Return the stripped content
            return None
        
        def get_initial_state_facts_from_pddl_instance(self, pddl_instance: str) -> list[str]:
            """
            Extracts initial state facts from a PDDL instance.
            This method should be implemented to return a list of initial state facts in PDDL format.
            """
            init_content = self.get_section_content(":init", pddl_instance)
            if init_content is None:
                raise ValueError("Invalid PDDL instance format: Could not find :init section content.")
            
            init_facts_raw = re.findall(self.FACT_PATTERN, init_content)
            init_facts_pddl = [re.sub(r"\s+", " ", fact).strip() for fact in init_facts_raw if fact.strip()]
            init_facts_nl = []
            for fact in init_facts_pddl:
                try:
                    init_facts_nl.append(self.translate_pddl_fact_to_natural_language(fact))
                except ValueError as e:
                    logger.warning(f"Skipping initial fact due to conversion error: {e} (Fact: '{fact}')")
            return init_facts_nl
        
        def get_goal_facts_from_pddl_instance(self, pddl_instance: str) -> list[str]:
            """
            Extracts goal facts from a PDDL instance.
            This method should be implemented to return a list of goal facts in PDDL format.
            """
            goal_content = self.get_section_content(":goal", pddl_instance)
            if goal_content is None:
                raise ValueError("Invalid PDDL instance format: Could not find :goal section content.")

            goal_facts_raw = []
            and_match = re.search(r"^\(and\s+(.*)\s*\)$", goal_content, re.DOTALL | re.IGNORECASE)
            if and_match:
                and_content = and_match.group(1).strip()
                goal_facts_raw = re.findall(self.FACT_PATTERN, and_content)
            else:
                stripped_goal_content = goal_content.strip()
                if stripped_goal_content and re.fullmatch(self.FACT_PATTERN, stripped_goal_content):
                    goal_facts_raw = [stripped_goal_content]
                elif not stripped_goal_content:
                    pass 

            goal_facts_pddl = [re.sub(r"\s+", " ", fact).strip() for fact in goal_facts_raw if fact.strip()]

            goal_facts_nl = []
            for fact in goal_facts_pddl:
                try:
                    goal_facts_nl.append(self.translate_pddl_fact_to_natural_language(fact))
                except ValueError as e:
                    logger.warning(f"Skipping goal fact due to conversion error: {e} (Fact: '{fact}')")

            return goal_facts_nl

        @abc.abstractmethod
        def translate_pddl_fact_to_natural_language(self, fact: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")

        @abc.abstractmethod
        def translate_pddl_plan_to_natural_language(self, pddl_plan: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")
        
        @abc.abstractmethod
        def translate_natural_language_plan_to_pddl(self, nl_plan: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")
        
        @property
        @abc.abstractmethod
        def _domain_description_in_natural_language(self) -> str:
            raise NotImplementedError("Subclasses must implement this property.")


    def __init__(self, domain : str, domain_file_path : str, instance_file_path : str):
        self._domain :str = domain
        self._domain_file_path : str = domain_file_path
        self._instance_file_path : str = instance_file_path
        self._is_longer_plan : bool = True if config.LONG_INSTANCES in self._instance_file_path else False
        self._id : int = int(re.search(instance_pattern, self._instance_file_path).group(1))
        self._paas_status : Optional[Task.PAAS_STATUS] = None # ok, error
        self._pddl_plan : Optional[str] = None
        self._type : Optional[Task.TYPE] = None # training, validation, test | None

        if self._domain == "blocksworld":
            self._domain_translator : Task.DomainTranslator = BlocksworldTranslator()
        else:
            raise ValueError(f"Unknown domain: {self._domain}. Supported domains are: blocksworld.")

    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented

        if self._domain != other._domain:
            return self._domain < other._domain

        if self._is_longer_plan != other._is_longer_plan:
            return not self._is_longer_plan

        self_match = instance_pattern.search(self._instance_file_path)
        other_match = instance_pattern.search(other._instance_file_path)
        if self_match and other_match:
            return int(self_match.group(1)) < int(other_match.group(1))
        else:
            return self._instance_file_path < other._instance_file_path

    def __str__(self):
        long_part = ", long" if self._is_longer_plan else ""
        type_part = f": {self._type.value}" if self._type else ""
        return f"Task {self._domain}, {self._id}{long_part}{type_part}"

    def __hash__(self):
        return hash((self._domain_file_path, self._instance_file_path))

    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self._instance_file_path == other._instance_file_path and self._domain_file_path == other._domain_file_path

    def get_task_components_in_natural_language(self, with_plan:bool = True) -> dict[str, str]:
        domain_description = self._domain_translator._domain_description_in_natural_language.strip()
        pddl_instance = self.read_instance()
        initial_state_facts_nl = self._domain_translator.get_initial_state_facts_from_pddl_instance(pddl_instance=pddl_instance)
        goal_facts_nl = self._domain_translator.get_goal_facts_from_pddl_instance(pddl_instance=pddl_instance)
        if self._pddl_plan:
            plan_nl = self._domain_translator.translate_pddl_plan_to_natural_language(pddl_plan=self._pddl_plan)
        else:
            if with_plan:
                raise ValueError("PDDL plan is not available, but 'with_plan' is set to True. Please ensure the task has a PDDL plan before requesting it.")
            else:
                plan_nl = ""
        return {
            "domain_description": domain_description,
            "initial_state_facts": initial_state_facts_nl,
            "goal_facts": goal_facts_nl,
            "plan": plan_nl
        }

    def get_chat(self, with_plan: bool = True, prompt_type : config.PROMPT_TYPE = config.PROMPT_TYPE.IO, **kwargs) -> list[dict[str, str]]:
        task_components_in_nl = self.get_task_components_in_natural_language(with_plan=with_plan)
        if prompt_type == config.PROMPT_TYPE.IO:
            # TODO: VERIFICAR SE A MODIFICAÇÃO NÃO CAUSOU ERROS
            initial_state_facts_str = "As initial conditions I have that: " + (", ".join(task_components_in_nl['initial_state_facts']))
            goal_facts_str = "My goal is to have that: " + (", ".join(task_components_in_nl['goal_facts']))      
            chat: list[dict[str, str]] = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{task_components_in_nl['domain_description']}\n{initial_state_facts_str}\n{goal_facts_str}"},
            ]       
            if with_plan:
                chat.append({"role": "assistant", "content": f"My plan is as follows:\n{config.TOKENS.PLAN_START.value}\n{task_components_in_nl['plan']}\n{config.TOKENS.PLAN_END.value}"})
            return chat
        elif prompt_type == config.PROMPT_TYPE.FEW_SHOT:
            few_shot = kwargs.get("few_shot", 1)

            # TODO: UNCOMMENT THIS LATER
            # few_shot_examples = get_few_shot_examples(few_shot=few_shot)
            
            # TODO: REMOVE THIS LATER
            few_shot_examples = get_few_shot_examples(few_shot=few_shot - 1)
            validation_tasks = get_tasks(
                filter_by_domain=self._domain,
                filter_by_type=Task.TYPE.VALIDATION,
                is_longer_plan=False,
                number_of_instances=1
            )
            validation_task = next(iter(validation_tasks), None)
            if validation_task is None:
                raise ValueError(f"No validation task found for domain '{self._domain}'.")
            validation_data = validation_task.get_task_components_in_natural_language(with_plan=True)
            few_shot_examples.append(
                {
                    "domain_description": validation_data['domain_description'],
                    "initial_state_facts": "\n".join(validation_data['initial_state_facts']),
                    "goal_facts": "\n".join(validation_data['goal_facts']),
                    "plan": validation_data['plan'],
                }
            )
            # TODO: REMOVE UNTIL HERE LATER

            examples = []
            initial_state_facts_str = "\n".join(task_components_in_nl['initial_state_facts'])
            goal_facts_str = "\n".join(task_components_in_nl['goal_facts'])
            for data in few_shot_examples:
                examples.append(f"""{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
{data['domain_description']}
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.INITIAL_STATE_START.value}
{data['initial_state_facts']}
{config.TOKENS.INITIAL_STATE_END.value}
{config.TOKENS.GOAL_START.value}
{data['goal_facts']}
{config.TOKENS.GOAL_END.value}
{config.TOKENS.PLAN_START.value}
{data['plan']}
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}\n""")
            
            examples_content = "\n".join(examples)
            
            content = f"""Your task is to find a plan for a Blocksworld problem based on the provided domain and instance. The following examples show the required output format. Your response should contain only the plan arround the tags "{config.TOKENS.PLAN_START.value}...{config.TOKENS.PLAN_END.value}".

{config.TOKENS.DOMAIN_START.value}
{task_components_in_nl['domain_description']}
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.INITIAL_STATE_START.value}
{initial_state_facts_str}
{config.TOKENS.INITIAL_STATE_END.value}
{config.TOKENS.GOAL_START.value}
{goal_facts_str}
{config.TOKENS.GOAL_END.value}

{examples_content}

Here is a checklist to help you with your task:
1) Do not provide any additional text or explanations outside the plan tags.
2) The plan must be in the same format as the examples above.
3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
"""
            chat = [
                {"role": "system", "content": "You are an expert in AI Planning."},
                {"role": "user", "content": content}
            ]
            
            # TODO: REMOVE THIS LATER
            filename_for_debuging_prompt = f"debugging_prompt.txt"
            if not os.path.exists(filename_for_debuging_prompt):
                with open(filename_for_debuging_prompt, "w", encoding='utf-8') as f:
                    f.write(content)
            # TODO: REMOVE UNTIL HERE LATER

            if with_plan:
                chat.append({"role": "assistant", "content": f"{config.TOKENS.PLAN_START.value}\n{task_components_in_nl['plan']}\n{config.TOKENS.PLAN_END.value}"})
            return chat
        elif prompt_type == config.PROMPT_TYPE.PDDL:
            from string import Template

            PDDL_TEMPLATE_PROMPT = Template(f"""Your task is to find a plan for a Blocksworld problem based on the provided domain and instance. The following examples show the required output format. Your response should contain only the plan arround the tags '{config.TOKENS.PLAN_START.value}...{config.TOKENS.PLAN_END.value}'. 
                                         
            {config.TOKENS.DOMAIN_START.value}
            $domain_description
            {config.TOKENS.DOMAIN_END.value}
            {config.TOKENS.INSTANCE_START.value}
            $instance_content
            {config.TOKENS.INSTANCE_END.value}
            
            {config.TOKENS.EXAMPLE_START.value}
            {config.TOKENS.DOMAIN_START.value}
            (define (domain gripper-strips)
            (:predicates (room ?r)
                    (ball ?b)
                    (gripper ?g)
                    (at-robby ?r)
                    (at ?b ?r)
                    (free ?g)
                    (carry ?o ?g))

            (:action move
                :parameters  (?from ?to)
                :precondition (and  (room ?from) (room ?to) (at-robby ?from))
                :effect (and  (at-robby ?to)
                        (not (at-robby ?from))))

            (:action pick
                :parameters (?obj ?room ?gripper)
                :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                            (at ?obj ?room) (at-robby ?room) (free ?gripper))
                :effect (and (carry ?obj ?gripper)
                        (not (at ?obj ?room))
                        (not (free ?gripper))))

            (:action drop
                :parameters  (?obj  ?room ?gripper)
                :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                            (carry ?obj ?gripper) (at-robby ?room))
                :effect (and (at ?obj ?room)
                        (free ?gripper)
                        (not (carry ?obj ?gripper)))))
            {config.TOKENS.DOMAIN_END.value}
            {config.TOKENS.INSTANCE_START.value}
            (define (problem strips-gripper-x-20)
            (:domain gripper-strips)
            (:objects rooma roomb ball42 ball41 ball40 ball39 ball38 ball37
                        ball36 ball35 ball34 ball33 ball32 ball31 ball30 ball29 ball28
                        ball27 ball26 ball25 ball24 ball23 ball22 ball21 ball20 ball19
                        ball18 ball17 ball16 ball15 ball14 ball13 ball12 ball11 ball10
                        ball9 ball8 ball7 ball6 ball5 ball4 ball3 ball2 ball1 left right)
            (:init (room rooma)
                    (room roomb)
                    (ball ball42)
                    (ball ball41)
                    (ball ball40)
                    (ball ball39)
                    (ball ball38)
                    (ball ball37)
                    (ball ball36)
                    (ball ball35)
                    (ball ball34)
                    (ball ball33)
                    (ball ball32)
                    (ball ball31)
                    (ball ball30)
                    (ball ball29)
                    (ball ball28)
                    (ball ball27)
                    (ball ball26)
                    (ball ball25)
                    (ball ball24)
                    (ball ball23)
                    (ball ball22)
                    (ball ball21)
                    (ball ball20)
                    (ball ball19)
                    (ball ball18)
                    (ball ball17)
                    (ball ball16)
                    (ball ball15)
                    (ball ball14)
                    (ball ball13)
                    (ball ball12)
                    (ball ball11)
                    (ball ball10)
                    (ball ball9)
                    (ball ball8)
                    (ball ball7)
                    (ball ball6)
                    (ball ball5)
                    (ball ball4)
                    (ball ball3)
                    (ball ball2)
                    (ball ball1)
                    (at-robby rooma)
                    (free left)
                    (free right)
                    (at ball42 rooma)
                    (at ball41 rooma)
                    (at ball40 rooma)
                    (at ball39 rooma)
                    (at ball38 rooma)
                    (at ball37 rooma)
                    (at ball36 rooma)
                    (at ball35 rooma)
                    (at ball34 rooma)
                    (at ball33 rooma)
                    (at ball32 rooma)
                    (at ball31 rooma)
                    (at ball30 rooma)
                    (at ball29 rooma)
                    (at ball28 rooma)
                    (at ball27 rooma)
                    (at ball26 rooma)
                    (at ball25 rooma)
                    (at ball24 rooma)
                    (at ball23 rooma)
                    (at ball22 rooma)
                    (at ball21 rooma)
                    (at ball20 rooma)
                    (at ball19 rooma)
                    (at ball18 rooma)
                    (at ball17 rooma)
                    (at ball16 rooma)
                    (at ball15 rooma)
                    (at ball14 rooma)
                    (at ball13 rooma)
                    (at ball12 rooma)
                    (at ball11 rooma)
                    (at ball10 rooma)
                    (at ball9 rooma)
                    (at ball8 rooma)
                    (at ball7 rooma)
                    (at ball6 rooma)
                    (at ball5 rooma)
                    (at ball4 rooma)
                    (at ball3 rooma)
                    (at ball2 rooma)
                    (at ball1 rooma)
                    (gripper left)
                    (gripper right))
            (:goal (and (at ball42 roomb)
                        (at ball41 roomb)
                        (at ball40 roomb)
                        (at ball39 roomb)
                        (at ball38 roomb)
                        (at ball37 roomb)
                        (at ball36 roomb)
                        (at ball35 roomb)
                        (at ball34 roomb)
                        (at ball33 roomb)
                        (at ball32 roomb)
                        (at ball31 roomb)
                        (at ball30 roomb)
                        (at ball29 roomb)
                        (at ball28 roomb)
                        (at ball27 roomb)
                        (at ball26 roomb)
                        (at ball25 roomb)
                        (at ball24 roomb)
                        (at ball23 roomb)
                        (at ball22 roomb)
                        (at ball21 roomb)
                        (at ball20 roomb)
                        (at ball19 roomb)
                        (at ball18 roomb)
                        (at ball17 roomb)
                        (at ball16 roomb)
                        (at ball15 roomb)
                        (at ball14 roomb)
                        (at ball13 roomb)
                        (at ball12 roomb)
                        (at ball11 roomb)
                        (at ball10 roomb)
                        (at ball9 roomb)
                        (at ball8 roomb)
                        (at ball7 roomb)
                        (at ball6 roomb)
                        (at ball5 roomb)
                        (at ball4 roomb)
                        (at ball3 roomb)
                        (at ball2 roomb)
                        (at ball1 roomb))))
            {config.TOKENS.INSTANCE_END.value}
            {config.TOKENS.PLAN_START.value}
            (pick ball9 rooma right)
            (move rooma roomb)
            (drop ball9 roomb right)
            (move roomb rooma)
            (pick ball5 rooma right)
            (move rooma roomb)
            (drop ball5 roomb right)
            (move roomb rooma)
            (pick ball7 rooma right)
            (move rooma roomb)
            (drop ball7 roomb right)
            (move roomb rooma)
            (pick ball6 rooma right)
            (move rooma roomb)
            (drop ball6 roomb right)
            (move roomb rooma)
            (pick ball42 rooma right)
            (move rooma roomb)
            (drop ball42 roomb right)
            (move roomb rooma)
            (pick ball40 rooma right)
            (move rooma roomb)
            (drop ball40 roomb right)
            (move roomb rooma)
            (pick ball4 rooma right)
            (move rooma roomb)
            (drop ball4 roomb right)
            (move roomb rooma)
            (pick ball37 rooma right)
            (move rooma roomb)
            (drop ball37 roomb right)
            (move roomb rooma)
            (pick ball36 rooma right)
            (move rooma roomb)
            (drop ball36 roomb right)
            (move roomb rooma)
            (pick ball35 rooma right)
            (move rooma roomb)
            (drop ball35 roomb right)
            (move roomb rooma)
            (pick ball17 rooma right)
            (move rooma roomb)
            (drop ball17 roomb right)
            (move roomb rooma)
            (pick ball20 rooma right)
            (move rooma roomb)
            (drop ball20 roomb right)
            (move roomb rooma)
            (pick ball15 rooma right)
            (move rooma roomb)
            (drop ball15 roomb right)
            (move roomb rooma)
            (pick ball14 rooma right)
            (move rooma roomb)
            (drop ball14 roomb right)
            (move roomb rooma)
            (pick ball13 rooma right)
            (move rooma roomb)
            (drop ball13 roomb right)
            (move roomb rooma)
            (pick ball16 rooma right)
            (move rooma roomb)
            (drop ball16 roomb right)
            (move roomb rooma)
            (pick ball8 rooma right)
            (move rooma roomb)
            (drop ball8 roomb right)
            (move roomb rooma)
            (pick ball25 rooma right)
            (move rooma roomb)
            (drop ball25 roomb right)
            (move roomb rooma)
            (pick ball2 rooma right)
            (move rooma roomb)
            (drop ball2 roomb right)
            (move roomb rooma)
            (pick ball26 rooma right)
            (move rooma roomb)
            (drop ball26 roomb right)
            (move roomb rooma)
            (pick ball22 rooma right)
            (move rooma roomb)
            (drop ball22 roomb right)
            (move roomb rooma)
            (pick ball28 rooma right)
            (move rooma roomb)
            (drop ball28 roomb right)
            (move roomb rooma)
            (pick ball12 rooma right)
            (move rooma roomb)
            (drop ball12 roomb right)
            (move roomb rooma)
            (pick ball10 rooma right)
            (move rooma roomb)
            (drop ball10 roomb right)
            (move roomb rooma)
            (pick ball19 rooma right)
            (move rooma roomb)
            (drop ball19 roomb right)
            (move roomb rooma)
            (pick ball21 rooma right)
            (move rooma roomb)
            (drop ball21 roomb right)
            (move roomb rooma)
            (pick ball39 rooma right)
            (move rooma roomb)
            (drop ball39 roomb right)
            (move roomb rooma)
            (pick ball23 rooma right)
            (move rooma roomb)
            (drop ball23 roomb right)
            (move roomb rooma)
            (pick ball24 rooma right)
            (move rooma roomb)
            (drop ball24 roomb right)
            (move roomb rooma)
            (pick ball41 rooma right)
            (move rooma roomb)
            (drop ball41 roomb right)
            (move roomb rooma)
            (pick ball29 rooma right)
            (move rooma roomb)
            (drop ball29 roomb right)
            (move roomb rooma)
            (pick ball38 rooma right)
            (move rooma roomb)
            (drop ball38 roomb right)
            (move roomb rooma)
            (pick ball27 rooma right)
            (move rooma roomb)
            (drop ball27 roomb right)
            (move roomb rooma)
            (pick ball32 rooma right)
            (move rooma roomb)
            (drop ball32 roomb right)
            (move roomb rooma)
            (pick ball3 rooma right)
            (move rooma roomb)
            (drop ball3 roomb right)
            (move roomb rooma)
            (pick ball11 rooma right)
            (move rooma roomb)
            (drop ball11 roomb right)
            (move roomb rooma)
            (pick ball1 rooma right)
            (move rooma roomb)
            (drop ball1 roomb right)
            (move roomb rooma)
            (pick ball30 rooma right)
            (move rooma roomb)
            (drop ball30 roomb right)
            (move roomb rooma)
            (pick ball18 rooma right)
            (move rooma roomb)
            (drop ball18 roomb right)
            (move roomb rooma)
            (pick ball34 rooma right)
            (move rooma roomb)
            (drop ball34 roomb right)
            (move roomb rooma)
            (pick ball31 rooma right)
            (move rooma roomb)
            (drop ball31 roomb right)
            (move roomb rooma)
            (pick ball33 rooma right)
            (move rooma roomb)
            (drop ball33 roomb right)
            {config.TOKENS.PLAN_END.value}
            {config.TOKENS.EXAMPLE_END.value}

            {config.TOKENS.EXAMPLE_START.value}
            {config.TOKENS.DOMAIN_START.value}
            (define (domain logistics-strips)
            (:requirements :strips)
            (:predicates 	(OBJ ?obj)
                        (TRUCK ?truck)
                            (LOCATION ?loc)
                    (AIRPLANE ?airplane)
                            (CITY ?city)
                            (AIRPORT ?airport)
                    (at ?obj ?loc)
                    (in ?obj1 ?obj2)
                    (in-city ?obj ?city))

            (:action LOAD-TRUCK
            :parameters
            (?obj
                ?truck
                ?loc)
            :precondition
            (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
            (at ?truck ?loc) (at ?obj ?loc))
            :effect
            (and (not (at ?obj ?loc)) (in ?obj ?truck)))

            (:action LOAD-AIRPLANE
            :parameters
            (?obj
                ?airplane
                ?loc)
            :precondition
            (and (OBJ ?obj) (AIRPLANE ?airplane) (LOCATION ?loc)
            (at ?obj ?loc) (at ?airplane ?loc))
            :effect
            (and (not (at ?obj ?loc)) (in ?obj ?airplane)))

            (:action UNLOAD-TRUCK
            :parameters
            (?obj
                ?truck
                ?loc)
            :precondition
            (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
                    (at ?truck ?loc) (in ?obj ?truck))
            :effect
            (and (not (in ?obj ?truck)) (at ?obj ?loc)))

            (:action UNLOAD-AIRPLANE
            :parameters
            (?obj
                ?airplane
                ?loc)
            :precondition
            (and (OBJ ?obj) (AIRPLANE ?airplane) (LOCATION ?loc)
                    (in ?obj ?airplane) (at ?airplane ?loc))
            :effect
            (and (not (in ?obj ?airplane)) (at ?obj ?loc)))

            (:action DRIVE-TRUCK
            :parameters
            (?truck
                ?loc-from
                ?loc-to
                ?city)
            :precondition
            (and (TRUCK ?truck) (LOCATION ?loc-from) (LOCATION ?loc-to) (CITY ?city)
            (at ?truck ?loc-from)
            (in-city ?loc-from ?city)
            (in-city ?loc-to ?city))
            :effect
            (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to)))

            (:action FLY-AIRPLANE
            :parameters
            (?airplane
                ?loc-from
                ?loc-to)
            :precondition
            (and (AIRPLANE ?airplane) (AIRPORT ?loc-from) (AIRPORT ?loc-to)
                (at ?airplane ?loc-from))
            :effect
            (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to)))
            )
            {config.TOKENS.DOMAIN_END.value}
            {config.TOKENS.INSTANCE_START.value}
            (define (problem strips-log-y-5)
            (:domain logistics-strips)
            (:objects package5 package4 package3 package2 package1 city8
                        city7 city6 city5 city4 city3 city2 city1 truck15 truck14
                        truck13 truck12 truck11 truck10 truck9 truck8 truck7 truck6
                        truck5 truck4 truck3 truck2 truck1 plane1 city8-2 city8-1
                        city7-2 city7-1 city6-2 city6-1 city5-2 city5-1 city4-2
                        city4-1 city3-2 city3-1 city2-2 city2-1 city1-2 city1-1
                        city8-3 city7-3 city6-3 city5-3 city4-3 city3-3 city2-3
                        city1-3)
            (:init (obj package5)
                    (obj package4)
                    (obj package3)
                    (obj package2)
                    (obj package1)
                    (city city8)
                    (city city7)
                    (city city6)
                    (city city5)
                    (city city4)
                    (city city3)
                    (city city2)
                    (city city1)
                    (truck truck15)
                    (truck truck14)
                    (truck truck13)
                    (truck truck12)
                    (truck truck11)
                    (truck truck10)
                    (truck truck9)
                    (truck truck8)
                    (truck truck7)
                    (truck truck6)
                    (truck truck5)
                    (truck truck4)
                    (truck truck3)
                    (truck truck2)
                    (truck truck1)
                    (airplane plane1)
                    (location city8-2)
                    (location city8-1)
                    (location city7-2)
                    (location city7-1)
                    (location city6-2)
                    (location city6-1)
                    (location city5-2)
                    (location city5-1)
                    (location city4-2)
                    (location city4-1)
                    (location city3-2)
                    (location city3-1)
                    (location city2-2)
                    (location city2-1)
                    (location city1-2)
                    (location city1-1)
                    (airport city8-3)
                    (location city8-3)
                    (airport city7-3)
                    (location city7-3)
                    (airport city6-3)
                    (location city6-3)
                    (airport city5-3)
                    (location city5-3)
                    (airport city4-3)
                    (location city4-3)
                    (airport city3-3)
                    (location city3-3)
                    (airport city2-3)
                    (location city2-3)
                    (airport city1-3)
                    (location city1-3)
                    (in-city city8-3 city8)
                    (in-city city8-2 city8)
                    (in-city city8-1 city8)
                    (in-city city7-3 city7)
                    (in-city city7-2 city7)
                    (in-city city7-1 city7)
                    (in-city city6-3 city6)
                    (in-city city6-2 city6)
                    (in-city city6-1 city6)
                    (in-city city5-3 city5)
                    (in-city city5-2 city5)
                    (in-city city5-1 city5)
                    (in-city city4-3 city4)
                    (in-city city4-2 city4)
                    (in-city city4-1 city4)
                    (in-city city3-3 city3)
                    (in-city city3-2 city3)
                    (in-city city3-1 city3)
                    (in-city city2-3 city2)
                    (in-city city2-2 city2)
                    (in-city city2-1 city2)
                    (in-city city1-3 city1)
                    (in-city city1-2 city1)
                    (in-city city1-1 city1)
                    (at plane1 city3-3)
                    (at truck15 city8-2)
                    (at truck14 city7-2)
                    (at truck13 city6-1)
                    (at truck12 city5-2)
                    (at truck11 city4-1)
                    (at truck10 city3-2)
                    (at truck9 city2-1)
                    (at truck8 city1-2)
                    (at truck7 city6-1)
                    (at truck6 city3-2)
                    (at truck5 city1-3)
                    (at truck4 city4-3)
                    (at truck3 city1-3)
                    (at truck2 city7-1)
                    (at truck1 city8-1)
                    (at package5 city3-2)
                    (at package4 city5-1)
                    (at package3 city1-1)
                    (at package2 city5-2)
                    (at package1 city2-1))
            (:goal (and (at package5 city6-3)
                        (at package4 city5-3)
                        (at package3 city8-3)
                        (at package2 city4-3)
                        (at package1 city6-3))))
            {config.TOKENS.INSTANCE_END.value}
            {config.TOKENS.PLAN_START.value}
            (load-truck package2 truck12 city5-2)
            (drive-truck truck12 city5-2 city5-1 city5)
            (load-truck package1 truck9 city2-1)
            (drive-truck truck9 city2-1 city2-3 city2)
            (load-truck package4 truck12 city5-1)
            (drive-truck truck12 city5-1 city5-3 city5)
            (unload-truck package4 truck12 city5-3)
            (load-truck package5 truck10 city3-2)
            (drive-truck truck10 city3-2 city3-3 city3)
            (unload-truck package2 truck12 city5-3)
            (unload-truck package1 truck9 city2-3)
            (unload-truck package5 truck10 city3-3)
            (load-airplane package5 plane1 city3-3)
            (fly-airplane plane1 city3-3 city6-3)
            (unload-airplane package5 plane1 city6-3)
            (drive-truck truck3 city1-3 city1-1 city1)
            (load-truck package3 truck3 city1-1)
            (drive-truck truck3 city1-1 city1-3 city1)
            (fly-airplane plane1 city6-3 city5-3)
            (load-airplane package2 plane1 city5-3)
            (fly-airplane plane1 city5-3 city6-3)
            (unload-truck package3 truck3 city1-3)
            (fly-airplane plane1 city6-3 city4-3)
            (unload-airplane package2 plane1 city4-3)
            (fly-airplane plane1 city4-3 city2-3)
            (load-airplane package1 plane1 city2-3)
            (fly-airplane plane1 city2-3 city6-3)
            (unload-airplane package1 plane1 city6-3)
            (fly-airplane plane1 city6-3 city1-3)
            (load-airplane package3 plane1 city1-3)
            (fly-airplane plane1 city1-3 city8-3)
            (unload-airplane package3 plane1 city8-3)
            {config.TOKENS.PLAN_END.value}
            {config.TOKENS.EXAMPLE_END.value}

            Here is a checklist to help you with your task:
            1) Do not provide any additional text or explanations outside the plan tags.
            2) The plan must be in the same format as the examples above.
            3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
            4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
            5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.""")

            instance = self.read_instance()
            domain = self.read_domain()
            content = PDDL_TEMPLATE_PROMPT.substitute(
                domain_description=domain,
                instance_content=instance
            )
            chat = [
                {"role": "system", "content": "You are an expert in AI Planning."},
                {"role": "user", "content": content}
            ]
            if with_plan:
                chat.append({"role": "assistant", "content": f"{config.TOKENS.PLAN_START.value}\n{self._pddl_plan}\n{config.TOKENS.PLAN_END.value}"})
            return chat
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}. Supported types are: {list(config.PROMPT_TYPE)}.")

    def get_prompt_metadata(self, prompt_type: config.PROMPT_TYPE = config.PROMPT_TYPE.IO, **kwargs) -> dict[str, any]:
        if prompt_type == config.PROMPT_TYPE.IO:
            return {}
        elif prompt_type == config.PROMPT_TYPE.FEW_SHOT:
            few_shot = kwargs.get("few_shot", 1)
            return {"few_shot": few_shot}
        elif prompt_type == config.PROMPT_TYPE.PDDL:
            return {}
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}. Supported types are: {list(config.PROMPT_TYPE)}.")

    def to_json(self):
        try:
            data = {
                "domain": self._domain,
                "id": self._id,
                "domain_file_path": self._domain_file_path,
                "instance_file_path": self._instance_file_path,
                "paas_status": self._paas_status.value if self._paas_status else None,
                "pddl_plan": self._pddl_plan,
                "is_longer_plan": self._is_longer_plan,
                "type": self._type.value if self._type else None,
            }
        except (NotImplementedError, AssertionError, Exception) as e:
            logger.error(f"Error generating prompt for task {self._id}: {e}")
            raise e

        return json.dumps(data, ensure_ascii=False)


    def from_json(self, json_obj):
        # --- Basic Fields ---
        # id is derived in __init__, but we can load it from JSON if present
        json_id = json_obj.get("id")
        if json_id is not None:
            try:
                self._id = int(json_id)
            except (ValueError, TypeError):
                logger.error(f"Invalid id value in JSON: '{json_id}'. Expected integer.")
                # Decide if this should raise an error or just log
                raise ValueError(f"Invalid id value in JSON: '{json_id}'")

        # --- Optional Fields ---
        self._pddl_plan = json_obj.get("pddl_plan", None)

        # --- Enum Fields ---
        # Use correct internal field names (_pddl_status, _type)
        for field_name, json_key, enum_type in [
            ("_paas_status", "paas_status", Task.PAAS_STATUS),
            ("_type", "type", Task.TYPE)
        ]:
            json_value = json_obj.get(json_key)
            if json_value is not None:
                if not isinstance(json_value, str):
                    msg = f"Expected string for {json_key}, but got {type(json_value)}"
                    logger.error(msg)
                    raise ValueError(msg)
                try:
                    setattr(self, field_name, enum_type(json_value.strip()))
                except (ValueError, KeyError):
                    msg = f"Invalid {json_key} value in JSON: '{json_value}'"
                    logger.error(msg)
                    raise ValueError(msg)
                    
            else:
                setattr(self, field_name, None)

    def process_paas_response(self, response):
        plan = ""
        status = response.get("status", "error")
        if status == "ok":
            plan = response["result"]["output"]["sas_plan"]
        
        self._paas_status = Task.PAAS_STATUS(status) if status in [e.value for e in Task.PAAS_STATUS] else None
        self._pddl_plan = plan

    def read_instance(self):
        with lock and open(self._instance_file_path, "r", encoding='utf-8') as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self._domain_file_path, "r", encoding='utf-8') as f:
            domain_content = f.read()
        return domain_content

class BlocksworldTranslator(Task.DomainTranslator):
        def __init__(self):
            self._color_map = {
            "a": "red",
            "b": "blue",
            "c": "yellow",
            "d": "orange",
            "e": "green",
            "f": "purple",
            "g": "pink",
            "h": "brown",
            "i": "gray",
            "j": "cyan",
            "k": "magenta",
            "l": "lime",
            "m": "navy",
            "n": "teal",
            "o": "coral",
            "p": "salmon",
            "q": "gold",
            "r": "khaki",
            "s": "lavender",
            "t": "plum",
            "u": "peach",
            "v": "tan",
            "w": "beige",
            "x": "ivory",
            "y": "mint",
            "z": "pearl",
        }

        def translate_pddl_fact_to_natural_language(self, pddl_fact:str) -> str:
            """
                Converts a PDDL fact into natural language, using color mapping.

                Args:
                    fact: str - a line of PDDL representing a fact (e.g., "(on a b)", "(clear c)").

                Returns:
                    str - The natural language representation of the fact.

                Raises:
                    ValueError: If the fact format is unknown or if a block identifier
                                cannot be mapped to a color using self._color_map.
            """
            # print(f"fact: @{fact}@")
            if not pddl_fact:
                return ""

            # Regex patterns for different fact types
            patterns = {
                "handempty": re.compile(r"^\(\s*handempty\s*\)$"),
                "holding": re.compile(r"^\(\s*holding\s+(\w+)\s*\)$"),
                "clear": re.compile(r"^\(\s*clear\s+(\w+)\s*\)$"),
                "ontable": re.compile(r"^\(\s*ontable\s+(\w+)\s*\)$"),
                "on": re.compile(r"^\(\s*on\s+(\w+)\s+(\w+)\s*\)$"),
            }

            match = patterns["handempty"].match(pddl_fact)
            if match:
                return "the hand is empty"

            match = patterns["holding"].match(pddl_fact)
            if match:
                block = match.group(1)
                if block not in self._color_map:
                    raise ValueError(f"Unknown block '{block}' in fact: {pddl_fact}")
                return f"you are holding the {self._color_map[block]} block"

            match = patterns["clear"].match(pddl_fact)
            if match:
                block = match.group(1)
                if block not in self._color_map:
                    raise ValueError(f"Unknown block '{block}' in fact: {pddl_fact}")
                return f"the {self._color_map[block]} block is clear"

            match = patterns["ontable"].match(pddl_fact)
            if match:
                block = match.group(1)
                if block not in self._color_map:
                    raise ValueError(f"Unknown block '{block}' in fact: {pddl_fact}")
                return f"the {self._color_map[block]} block is on the table"

            match = patterns["on"].match(pddl_fact)
            if match:
                block1 = match.group(1)
                block2 = match.group(2)
                if block1 not in self._color_map:
                    raise ValueError(f"Unknown block '{block1}' in fact: {pddl_fact}")
                if block2 not in self._color_map:
                    raise ValueError(f"Unknown block '{block2}' in fact: {pddl_fact}")
                return f"the {self._color_map[block1]} block is on top of the {self._color_map[block2]} block"
            # If none of the patterns matched
            raise ValueError(f"Unknown or malformed fact format: {pddl_fact}")


        def translate_pddl_plan_to_natural_language(self, pddl_plan:str) -> str:
            """
            Converts a PDDL plan into natural language, using color mapping.

            Args:
                plan: str - PDDL plan string, with actions potentially separated by newlines or semicolons.

            Returns:
                str - The natural language representation of the plan, with actions separated by newlines.

            Raises:
                ValueError: If an action format is unknown or if a block identifier
                    cannot be mapped to a color using self._color_map.
            """
            nl_actions = []
            # Regex to capture action name and arguments (1 or 2 blocks)
            action_pattern = re.compile(r"^\(\s*([\w-]+)\s+(\w+)(?:\s+(\w+))?\s*\)$")

            action_templates = {
                "unstack": "unstack the {color1} block from the {color2} block",
                "pick-up": "pick up the {color1} block",
                "stack": "stack the {color1} block on top of the {color2} block",
                "put-down": "put down the {color1} block",
            }

            # Normalize line endings, remove comments, split into lines
            lines = pddl_plan.split(";")[0].replace("\r\n", "\n").strip().split("\n")

            for line in lines:
                action_str = line.strip().lower()
                if not action_str:
                    continue # Skip empty lines

                match = action_pattern.match(action_str)
                if not match:
                    raise ValueError(f"Unknown or malformed action format: {action_str}")

                action_name, block1_id, block2_id = match.groups() # block2_id might be None

                if action_name not in action_templates:
                    raise ValueError(f"Unknown action type '{action_name}' in action: {action_str}")

                # Validate and map block IDs to colors
                if block1_id not in self._color_map:
                    raise ValueError(f"Unknown block '{block1_id}' in action: {action_str}")
                color1 = self._color_map[block1_id]

                format_args = {"color1": color1}

                # Handle actions requiring two blocks ("unstack", "stack")
                if action_name in ["unstack", "stack"]:
                    if not block2_id: # Error: Missing second block
                        raise ValueError(f"Action '{action_name}' requires two blocks, but found only one in: {action_str}")
                    # Second block exists, validate and add it
                    if block2_id not in self._color_map:
                        raise ValueError(f"Unknown block '{block2_id}' in action: {action_str}")
                    format_args["color2"] = self._color_map[block2_id]

                # Handle actions requiring one block ("pick-up", "put-down")
                elif action_name in ["pick-up", "put-down"]:
                    if block2_id: # Error: Unexpected second block
                        raise ValueError(f"Action '{action_name}' requires one block, but found two in: {action_str}")
                    # Correct case for one-block actions: block2_id is None, nothing else needed.


                try:
                    nl_action = action_templates[action_name].format(**format_args)
                    nl_actions.append(nl_action)
                except KeyError as e:
                # This might happen if the template expects color2 but it wasn't provided
                    raise ValueError(f"Formatting error for action '{action_name}'. Missing argument: {e}. Action: {action_str}")


            return "\n".join(nl_actions)

        def translate_natural_language_plan_to_pddl(self, nl_plan:str) -> str:
            """
            Converts a natural language plan into PDDL format.

            Args:
                plan: A string containing actions in natural language, separated by newlines or semicolons.

            Returns:
                A string representing the plan in PDDL format.

            Raises:
                ValueError: If an unknown or malformed action is encountered.
            """
            pddl_actions = []
            action_mapping = {
                "pick up": "pick-up",
                "put down": "put-down",
                "unstack": "unstack",
                "stack": "stack"
            }

            # Regex to match action and extract block IDs
            pick_up_pattern = re.compile(r"^pick up (\w+)$")
            put_down_pattern = re.compile(r"^put down (\w+)$")
            unstack_pattern = re.compile(r"^unstack (\w+)(?: from)? (\w+)$")
            stack_pattern = re.compile(r"^stack (\w+)(?: (?:on top of|on|over) )?(\w+)$")

            for line in nl_plan.replace(";", "\n").strip().split("\n"):
                action_str = line.strip().lower()
                if not action_str:
                    continue
                action_str = action_str.strip().replace("the ", "").replace(" block", "")

                pick_up_match = pick_up_pattern.match(action_str)
                put_down_match = put_down_pattern.match(action_str)
                unstack_match = unstack_pattern.match(action_str)
                stack_match = stack_pattern.match(action_str)
                if pick_up_match:
                    block1_color = pick_up_match.group(1)
                    id1 = next((k for k, v in self._color_map.items() if v == block1_color), None)
                    if id1 is None:
                        raise ValueError(f"Unknown block color '{block1_color}' in action: {action_str}")
                    pddl_actions.append(f"(pick-up {id1})")
                elif put_down_match:
                    block1_color = put_down_match.group(1)
                    id1 = next((k for k, v in self._color_map.items() if v == block1_color), None)
                    if id1 is None:
                        raise ValueError(f"Unknown block color '{block1_color}' in action: {action_str}")
                    pddl_actions.append(f"(put-down {id1})")
                elif unstack_match:
                    block1_color = unstack_match.group(1)
                    block2_color = unstack_match.group(2)
                    id1 = next((k for k, v in self._color_map.items() if v == block1_color), None)
                    id2 = next((k for k, v in self._color_map.items() if v == block2_color), None)
                    if id1 is None or id2 is None:
                        raise ValueError(f"Unknown block colors '{block1_color}' or '{block2_color}' in action: {action_str}")
                    pddl_actions.append(f"(unstack {id1} {id2})")
                elif stack_match:
                    block1_color = stack_match.group(1)
                    block2_color = stack_match.group(2)
                    id1 = next((k for k, v in self._color_map.items() if v == block1_color), None)
                    id2 = next((k for k, v in self._color_map.items() if v == block2_color), None)
                    if id1 is None or id2 is None:
                        raise ValueError(f"Unknown block colors '{block1_color}' or '{block2_color}' in action: {action_str}")
                    pddl_actions.append(f"(stack {id1} {id2})")
                else:
                    raise ValueError(f"Unknown or malformed action: {action_str}")


            # Join actions into a single PDDL string
            pddl_plan = "\n".join(pddl_actions)
            return pddl_plan

        @property
        def _domain_description_in_natural_language(self) -> str:

            return """I am playing with a set of blocks where I need to arrange the blocks into stacks.
Here are the actions that can be performed:
Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block
The following are the restrictions on the actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking is clear.
Once I put down or stack a block, my hand becomes empty.
Once you stack a block on top of a second block, the second block is no longer clear.\n"""


# Removed unused import 'Dataset'
# from datasets import Dataset
def get_tasks_from_domain_directory(domain: str) -> set[Task]:
    # Assuming config.RAW_DIR, config.DOMAIN_FILE_NAME, config.BASIC_INSTANCES, config.LONG_INSTANCES are defined
    domain_file_path = os.path.join(config.RAW_DIR, domain, config.DOMAIN_FILE_NAME)

    # Determine which instance directories to include
    instance_dirs = [
        os.path.join(config.RAW_DIR, domain, config.BASIC_INSTANCES), # training, validation, test instances
        os.path.join(config.RAW_DIR, domain, config.LONG_INSTANCES) # out of distribution instances
    ]
    # Collect tasks
    tasks:set[Task] = set()
    for instance_dir in instance_dirs:
        if not os.path.exists(instance_dir):
            raise ValueError(f"Instance directory not found: {instance_dir}")

        for file_name in os.listdir(instance_dir):
            if instance_pattern.search(file_name):
                tasks.add(
                    Task(
                        domain=domain,
                        domain_file_path=domain_file_path,
                        instance_file_path=os.path.join(instance_dir, file_name)
                    )
                )

    return tasks

DATASET: set[Task] = set()
def get_dataset() -> set[Task]:
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    return DATASET
    

def get_tasks(filter_by_domain: Optional[str] = None,  filter_by_type: Optional[Task.TYPE] = None, is_longer_plan:Optional[bool] = None, number_of_instances: Optional[int] = None) -> set[Task]:
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    tasks = DATASET
    # --- Filter by domain and type ---
    if filter_by_domain:
        tasks = {t for t in tasks if t._domain == filter_by_domain}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found for domain '{filter_by_domain}'.")
    if filter_by_type:
        tasks = {t for t in tasks if t._type == filter_by_type}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found for type '{filter_by_type}'.")
    # --- Filter by basic or long tasks ---
    if is_longer_plan is not None:
        tasks = {t for t in tasks if t._is_longer_plan == is_longer_plan}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found with is_longer_plan={is_longer_plan}.")
    # --- Limit number of instances ---
    if number_of_instances is not None and isinstance(number_of_instances, int):
        tasks = set(sorted(tasks)[:min(number_of_instances, len(tasks))])
        if len(tasks) == 0:
            raise ValueError(f"No tasks found after filtering.")
    return tasks

def get_task(domain_file_path: str, instance_file_path: str) -> Task:
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    for task in DATASET:
        if task._domain_file_path == domain_file_path and task._instance_file_path == instance_file_path:
            return task
    raise ValueError(f"Task with domain file path '{domain_file_path}' and instance file path '{instance_file_path}' not found.")

def load() -> None:
    jsonl_file_path = config.TASKS_DATASET_FILE_PATH
    global DATASET
    if not os.path.exists(jsonl_file_path):
        raise ValueError(f"JSONL file not found: {jsonl_file_path}")
    tasks = set()
    logger.info(f"Loading tasks from {jsonl_file_path}.")
    with open(jsonl_file_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                domain = json_obj.get("domain", None)
                instance_file_path = json_obj.get("instance_file_path", None)
                domain_file_path = json_obj.get("domain_file_path", None)
                assert domain, "Domain is not specified in the JSON object."
                assert instance_file_path, "Instance file path is not specified in the JSON object."
                assert domain_file_path, "Domain file path is not specified in the JSON object."
                task = Task(
                    domain,
                    domain_file_path,
                    instance_file_path
                )
                task.from_json(json_obj)
                tasks.add(task)
            except Exception as e:
                m = f"Error processing task from file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e
    DATASET = tasks
    logger.info(f"Loaded {len(DATASET)} tasks from {jsonl_file_path}.")

def save()-> None:
    jsonl_file_path = config.TASKS_DATASET_FILE_PATH
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    logger.info(f"Saving {len(DATASET)} tasks to {jsonl_file_path}.")
    with open(jsonl_file_path, "w", encoding='utf-8') as f:
        for task in sorted(DATASET):
            try:
                json_str = task.to_json() # Get the JSON string representation
                f.write(json_str + "\n") # Write the JSON string followed by a newline
            except Exception as e:
                m = f"Error saving task to file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e
    logger.info(f"Saved {len(DATASET)} tasks to {jsonl_file_path}.")

import numpy as np
def get_few_shot_examples(few_shot:int) -> list[dict[str, str]]:
    # TODO: LATER, CHANGE THIS FUNCTION TO USE ONLY TEST TASKS AND FROM OTHER DOMAINS
    # FOR NOW, WE HAVE A LIST OF SAMPLES THAT WE CAN USE AS FEW-SHOT EXAMPLES
    gripper_data = {
        "domain_description": """I have to plan how to move objects between rooms using a robot with grippers. The robot can move between rooms and pick up or drop objects using its grippers.
Here are the actions that can be performed:
Move the robot from one room to another room.
A robot pick up an object from a room.
A robot drop an object in a room.
The following are the restrictions on the actions:
A robot can move from one room to another room only if the robot is in the from-room.
Once the robot has moved from one room to another room, the robot is no longer in the from-room and is in the to-room.
A robot can pick up an object from a room only if the robot is in the room and the object is also in the same room.
A robot can pick up an object from a room only if the robot's gripper is free.
Once the robot has picked up an object from a room, the object is no longer in the room and is carried by the robot.
Once the robot has picked up an object from a room, the robot's gripper is no longer free.
A robot can drop an object in a room only if the robot is in the room and the robot is carrying the object.
Once the robot has dropped an object in a room, the object is in the room and is no longer carried by the robot.
Once the robot has dropped an object in a room, the robot's gripper is free.""",
        "initial_state_facts": """ball_1 is at room_1
ball_2 is at room_1
ball_3 is at room_2
ball_4 is at room_2
ball_5 is at room_1
ball_6 is at room_1
ball_7 is at room_2
ball_8 is at room_1
ball_9 is at room_1
robot_1 is at room_2
robot_2 is at room_2
robot_3 is at room_2
robot_4 is at room_2
robot_1's left_gripper_1 is free
robot_1's right_gripper_1 is free
robot_2's left_gripper_2 is free
robot_2's right_gripper_2 is free
robot_3's left_gripper_3 is free
robot_3's right_gripper_3 is free
robot_4's left_gripper_4 is free
robot_4's right_gripper_4 is free""",
        "goal_facts": """ball_1 is at room_2
ball_2 is at room_2
ball_3 is at room_1
ball_4 is at room_2
ball_5 is at room_2
ball_6 is at room_1
ball_7 is at room_2
ball_8 is at room_1
ball_9 is at room_1""",
        "plan": """robot_4 picks ball_3 at room_2 with left_gripper_4
move robot_4 from room_2 to room_1
robot_4 drops ball_3 at room_1 with left_gripper_4
robot_4 picks ball_1 at room_1 with left_gripper_4
move robot_4 from room_1 to room_2
robot_4 drops ball_1 at room_2 with left_gripper_4
move robot_1 from room_2 to room_1
robot_1 picks ball_2 at room_1 with left_gripper_1
robot_1 picks ball_5 at room_1 with right_gripper_1
move robot_1 from room_1 to room_2
robot_1 drops ball_2 at room_2 with left_gripper_1
robot_1 drops ball_5 at room_2 with right_gripper_1"""
    }
    childsnack_data = {
        "domain_description": """I have to plan how to make and serve sandwiches for a group of children, taking into account that some of them are allergic to gluten.
There are two types of sandwiches: regular and gluten-free.
Here are the actions that can be performed:
Make a gluten-free sandwich.
Make a regular sandwich.
Put a sandwich on a tray.
Serve a gluten-free sandwich to an allergic child.
Serve a regular sandwich to a child.
Move a tray between kitchen and tables.
The following are the restrictions on the actions:
We can make a gluten-free sandwich only if there is a bread at kitchen and the bread is gluten-free.
We can make a gluten-free sandwich only if there is a content at kitchen and the content is gluten-free.
Once we make a gluten-free sandwich, the bread and content are no longer at kitchen.
Once we make a gluten-free sandwich, the sandwich is at kitchen and is gluten-free.
We can make a regular sandwich only if there is a bread at kitchen.
We can make a regular sandwich only if there is a content at kitchen.
Once we make a regular sandwich, the bread and content are no longer at kitchen.
Once we make a regular sandwich, the sandwich is at kitchen.
We can put a sandwich on a tray only if the sandwich is at kitchen.
We can put a sandwich on a tray only if the tray is also at kitchen.
Once we put a sandwich on a tray, the sandwich is no longer at kitchen but is on the tray.
We can serve a gluten-free sandwich to an allergic child only if the child is allergic to gluten.
We can serve a gluten-free sandwich to an allergic child only if the sandwich is on a tray and the sandwich is gluten-free.
We can serve a gluten-free sandwich to an allergic child only if the child is waiting for the sandwich at the table.
We can serve a regular sandwich to a child only if the tray is at the table where the child is waiting.
Once we serve a gluten-free sandwich to an allergic child, the sandwich is no longer on the tray.
Once we serve a gluten-free sandwich to an allergic child, we say the child has been served.
We can serve a regular sandwich to a child only if the child is not allergic to gluten.
We can serve a regular sandwich to a child only if the child is waiting for the sandwich at the table.
We can serve a regular sandwich to a child only if the sandwich is on a tray and the tray is at the table where the child is waiting.
Once we serve a regular sandwich to a child, the sandwich is no longer on the tray.
Once we serve a regular sandwich to a child, we say the child has been served.
We can move a tray from from-place to to-place only if the tray is at from-place.
Once we move a tray from from-place to to-place, the tray is no longer at from-place but is at to-place.""",
        "initial_state_facts": """child_2 is allergic to gluten
tray_1 is at kitchen
tray_2 is at kitchen
tray_3 is at kitchen
bread_1 is at kitchen
bread_2 is at kitchen
bread_3 is at kitchen
content_1 is at kitchen
content_2 is at kitchen
content_3 is at kitchen
bread_2 is gluten-free
content_3 is gluten-free
child_1 is not allergic to gluten
child_3 is not allergic to gluten
sandwich_1 is not ready yet
sandwich_2 is not ready yet
sandwich_3 is not ready yet
sandwich_4 is not ready yet
sandwich_5 is not ready yet
sandwich_6 is not ready yet
child_1 is waiting for sandwich at table_2
child_2 is waiting for sandwich at table_2
child_3 is waiting for sandwich at table_2""",
        "goal_facts": """child_1 has been served
child_2 has been served
child_3 has been served""",
        "plan": """make a gluten-free sandwich_1 using bread_2 and content_3
put sandwich_1 on tray_3
move tray_3 from kitchen to table_2
use tray_3 to serve gluten-free sandwich_1 to child_2 at table_2
make a regular sandwich_6 using bread_1 and content_1
put sandwich_6 on tray_2
move tray_2 from kitchen to table_2
use tray_2 to serve regular sandwich_6 to child_1 at table_2
make a regular sandwich_5 using bread_3 and content_2
put sandwich_5 on tray_1
move tray_1 from kitchen to table_2
use tray_1 to serve regular sandwich_5 to child_3 at table_2"""
    }
    logistics_data = {
        "domain_description": """I have to plan logistics to transport packages within cities via trucks and between cities via airplanes. Locations within a city are directly connected (trucks can move between any two such locations), and so are the cities. In each city there is exactly one truck and each city has one location that serves as an airport.
Here are the actions that can be performed:
Load a package into a truck.
Load a package into an airplane.
Unload a package from a truck.
Unload a package from an airplane.
Drive a truck from one location to another location.
Fly an airplane from one city to another city.
The following are the restrictions on the actions:
A package can be loaded into a truck only if the package and the truck are in the same location.
Once a package is loaded into a truck, the package is not at the location and is in the truck.
A package can be loaded into an airplane only if the package and the airplane are in the same location.
Once a package is loaded into an airplane, the package is not at the location and is in the airplane.
A package can be unloaded from a truck only if the package is in the truck.
Once a package is unloaded from a truck, the package is not in the truck and is at the location of the truck.
A package can be unloaded from an airplane only if the package is in the airplane.
Once a package is unloaded from an airplane, the package is not in the airplane and is at the location of the airplane.
A truck can be driven from one location to another if the truck is at the from-location and both from-location and to-location are locations in the same city.
Once a truck is driven from one location to another, it is not at the from-location and is at the to-location.
An airplane can be flown from one city to another if the from-location and the to-location are airports and the airplane is at the from-location.
Once an airplane is flown from one city to another the airplane is not at the from-location and is at the to-location.""",
        "initial_state_facts": """location_0-0 is an airport
location_1-0 is an airport
airplane_0 is at location_0-0
airplane_1 is at location_0-0
airplane_2 is at location_1-0
package_0 is at location_1-0
package_1 is at location_0-1
package_2 is at location_1-0
package_3 is at location_0-0
package_4 is at location_0-1
truck_0 is at location_0-0
truck_1 is at location_1-1
location_0-0 is in the city city_0
location_0-1 is in the city city_0
location_1-0 is in the city city_1
location_1-1 is in the city city_1""",
        "goal_facts": """package_0 is at location_1-1
package_1 is at location_0-0
package_2 is at location_1-1
package_3 is at location_0-1
package_4 is at location_0-1""",
        "plan": """load package_3 into truck_0 at location_0-0
drive truck_1 from location_1-1 to location_1-0 in city_1
load package_2 into truck_1 at location_1-0
load package_0 into truck_1 at location_1-0
drive truck_1 from location_1-0 to location_1-1 in city_1
unload package_2 from truck_1 at location_1-1
unload package_0 from truck_1 at location_1-1
drive truck_0 from location_0-0 to location_0-1 in city_0
unload package_3 from truck_0 at location_0-1
load package_1 into truck_0 at location_0-1
drive truck_0 from location_0-1 to location_0-0 in city_0
unload package_1 from truck_0 at location_0-0"""
    }
    satellite_data = {
        "domain_description": """I have to plan how to operate satellites in space equipped with various instruments. The satellites can be turned to point in different directions, their instruments can be switched on or off, calibrated, and used to take images using specific modes.
Here are the actions that can be performed:
Turn a satellite to a direction.
Switch on an instrument on a satellite.
Switch off an instrument on a satellite.
Calibrate an instrument on a satellite by pointing it to a calibration target.
Take an image of a direction using an instrument with a specific mode.
The following are the restrictions on the actions:
A satellite can be turned from a previous direction to a new direction only if the satellite is currently pointing to the previous direction. Once turned, it is no longer pointing to the previous direction.
An instrument can be switched on on a satellite only if the instrument is on board and power-available. Once switched on, the instrument becomes power-on, is no longer power-available, and becomes uncalibrated.
An instrument can be switched off on a satellite only if the instrument is on board and power-on. Once switched off, the instrument becomes power-available and is no longer power-on.
An instrument can be calibrated on a satellite only if it is on board, power-on, and pointing to its calibration target. Once calibrated, the instrument is marked as calibrated.
To take an image, an instrument must be on board, power-on, calibrated, the satellite must be pointing to the required direction, and the instrument must support the specified mode.
Once an image is taken, the image is available.""",
                "initial_state_facts": """the calibration target of instrument_0 is star_2
the calibration target of instrument_1 is star_2
the calibration target of instrument_2 is ground_station_1
the calibration target of instrument_3 is ground_station_1
the calibration target of instrument_4 is ground_station_1
the calibration target of instrument_5 is ground_station_4
the calibration target of instrument_6 is ground_station_4
the calibration target of instrument_7 is ground_station_4
the calibration target of instrument_8 is ground_station_4
the calibration target of instrument_9 is star_2
instrument_0 is on board satellite_0
instrument_1 is on board satellite_1
instrument_2 is on board satellite_1
instrument_3 is on board satellite_2
instrument_4 is on board satellite_2
instrument_5 is on board satellite_3
instrument_6 is on board satellite_3
instrument_7 is on board satellite_4
instrument_8 is on board satellite_4
instrument_9 is on board satellite_4
satellite_0 is pointing to star_0
satellite_1 is pointing to ground_station_3
satellite_2 is pointing to star_6
satellite_3 is pointing to star_0
satellite_4 is pointing to star_2
satellite_0 is power-available
satellite_1 is power-available
satellite_2 is power-available
satellite_3 is power-available
satellite_4 is power-available
instrument_0 supports image_mode_0
instrument_0 supports infrared_mode_1
instrument_1 supports image_mode_0
instrument_1 supports infrared_mode_1
instrument_2 supports infrared_mode_1
instrument_3 supports image_mode_0
instrument_3 supports infrared_mode_1
instrument_4 supports infrared_mode_1
instrument_5 supports infrared_mode_1
instrument_6 supports image_mode_0
instrument_7 supports image_mode_0
instrument_7 supports infrared_mode_1
instrument_8 supports infrared_mode_1
instrument_9 supports image_mode_0
instrument_9 supports infrared_mode_1""",
                "goal_facts": """phenomenon_5 has image in infrared_mode_1
star_6 has image in infrared_mode_1
star_7 has image in infrared_mode_1
star_8 has image in infrared_mode_1
satellite_0 is pointing to star_6""",
        "plan": """switch on instrument_9 on satellite_4
calibrate instrument_9 on satellite_4 pointing to calibration target star_2
turn satellite_4 from phenomenon_5 to star_2
take image of phenomenon_5 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_4 from star_6 to phenomenon_5
take image of star_6 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_4 from star_7 to star_6
take image of star_7 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_4 from star_8 to star_7
take image of star_8 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_0 from star_6 to star_0"""
    }
    barman_data = {
        "domain_description": """I have to plan actions for a robotic bartender to prepare cocktails. The bartender has two hands and works with various containers and ingredients to mix and serve drinks.
Here are the actions that can be performed:
Grasp a container (shot or shaker) from the table.
Leave a container on the table.
Fill-shot with an ingredient from a dispenser.
Refill-shot with the same ingredient it contained before.
Empty a shot.
Clean a shot.
Pour the content of a shot into a clean shaker.
Pour the content of a shot into a used shaker that already contain some ingredient.
Empty a shaker.
Clean a shaker.
Shake the shaker to mix the ingredients.
Pour the content of a shaker into a shot.""",
        "initial_state_facts": """shaker_23 is clean
shot_295 is clean
cocktail_1 has ingredient_163 as its first ingredient
cocktail_1 has ingredient_383 as its second ingredient
dispenser_114 dispenses ingredient_163
dispenser_213 dispenses ingredient_383
shaker_23 is empty
shot_295 is empty
left_hand is empty
right_hand is empty
level_1 is the next level after level_0
level_2 is the next level after level_1
shaker_23 is on the table
shot_295 is on the table
shaker_23's zero fill level is at level_0
shaker_23's fill level is at level_0""",
        "goal_facts": """shot_295 contains cocktail_1""",
        "plan": """grasp the shaker_23 using left_hand
grasp the shot_295 using right_hand
leave the shaker_23 using left_hand
fill the shot_295 on right_hand with ingredient_163 using dispenser_114 when left_hand is empty
pour from shot_295 containing ingredient_163 to clean shaker_23 using right_hand from level_0 to level_1
clean the shot_295 on right_hand used for ingredient_163 when left_hand is empty
fill the shot_295 on right_hand with ingredient_383 using dispenser_213 when left_hand is empty
pour from shot_295 containing ingredient_383 to used shaker_23 using right_hand from level_1 to level_2
clean the shot_295 on right_hand used for ingredient_383 when left_hand is empty
grasp the shaker_23 using left_hand
leave the shot_295 using right_hand
shake shaker_23 on left_hand containing ingredient_163 and ingredient_383 to get cocktail_1 when right_hand is empty
pour from shaker_23 to shot_295 containing cocktail_1 using left_hand from level_2 to level_1"""}
    data = [
        gripper_data,
        logistics_data,
        childsnack_data,
        satellite_data,
        barman_data
    ]
    # rng = np.random.RandomState(config.RANDOM_SEED)
    # return rng.choice(data, size=few_shot, replace=False).tolist() 
    return data[:min(few_shot, len(data))]  # Return the first 'few_shot' examples or all if fewer than 'few_shot'