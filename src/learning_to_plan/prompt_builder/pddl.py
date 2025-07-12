from string import Template
from learning_to_plan import config

PDDL_TEMPLATE_PROMPT = Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. Your response should only contain the plan.
                                
{config.TOKENS.DOMAIN_START.value}
$domain_description
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance_content
{config.TOKENS.PROBLEM_END.value}

Here are some examples of plans in the same format as the one you should provide:
            
{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
(define (domain hanoi)
    (:requirements :strips)
    (:predicates
        (clear ?x)
        (on ?x ?y)
        (smaller ?x ?y)
    )

    (:action move
        :parameters (?disc ?from ?to)
        :precondition (and (smaller ?to ?disc)
            (on ?disc ?from)
            (clear ?disc)
            (clear ?to))
        :effect (and (clear ?from)
            (on ?disc ?to)
            (not (on ?disc ?from))
            (not (clear ?to)))
    )
)
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
(define (problem hanoi-4)
    (:domain hanoi)
    (:objects
        peg1 peg2 peg3 disc1 disc2 disc3 disc4
    )
    (:init
        (smaller peg1 disc1)
        (smaller peg1 disc2)
        (smaller peg1 disc3)
        (smaller peg1 disc4)
        (smaller peg2 disc1)
        (smaller peg2 disc2)
        (smaller peg2 disc3)
        (smaller peg2 disc4)
        (smaller peg3 disc1)
        (smaller peg3 disc2)
        (smaller peg3 disc3)
        (smaller peg3 disc4)
        (smaller disc2 disc1)
        (smaller disc3 disc1)
        (smaller disc4 disc1)
        (smaller disc3 disc2)
        (smaller disc4 disc2)
        (smaller disc4 disc3)
        (clear peg2)
        (clear peg3)
        (clear disc1)
        (on disc4 peg1)
        (on disc3 disc4)
        (on disc2 disc3)
        (on disc1 disc2)
    )
    (:goal
        (and
            (on disc4 peg3)
            (on disc3 disc4)
            (on disc2 disc3)
            (on disc1 disc2)
        )
    )
)
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.PLAN_START.value}
(move disc1 disc2 peg2)
(move disc2 disc3 peg3)
(move disc1 peg2 disc2)
(move disc3 disc4 peg2)
(move disc1 disc2 disc4)
(move disc2 peg3 disc3)
(move disc1 disc4 disc2)
(move disc4 peg1 peg3)
(move disc1 disc2 disc4)
(move disc2 disc3 peg1)
(move disc1 disc4 disc2)
(move disc3 peg2 disc4)
(move disc1 disc2 peg2)
(move disc2 peg1 disc3)
(move disc1 peg2 disc2)
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}

{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
(define (domain Storage-Propositional)
    (:requirements :typing)
    (:types
        hoist crate place area - object
        container depot - place
        storearea transitarea - area
    )

    (:predicates
        (clear ?s - storearea)
        (in ?x -
            (either storearea crate) ?p - place)
        (available ?h - hoist)
        (lifting ?h - hoist ?c - crate)
        (at ?h - hoist ?a - area)
        (on ?c - crate ?s - storearea)
        (connected ?a1 ?a2 - area)
        (compatible ?c1 ?c2 - crate)
    )

    (:action lift
        :parameters (?h - hoist ?c - crate ?a1 - storearea ?a2 - area ?p - place)
        :precondition (and
            (connected ?a1 ?a2)
            (at ?h ?a2)
            (available ?h)
            (on ?c ?a1)
            (in ?a1 ?p))
        :effect (and
            (not (on ?c ?a1))
            (clear ?a1)
            (not (available ?h))
            (lifting ?h ?c)
            (not (in ?c ?p)))
    )

    (:action drop
        :parameters (?h - hoist ?c - crate ?a1 - storearea ?a2 - area ?p - place)
        :precondition (and
            (connected ?a1 ?a2)
            (at ?h ?a2)
            ( lifting ?h ?c)
            (clear ?a1)
            (in ?a1 ?p))
        :effect (and
            (not (lifting ?h ?c))
            (available ?h)
            (not (clear ?a1))
            (on ?c ?a1)
            (in ?c ?p))
    )

    (:action move
        :parameters (?h - hoist ?from ?to - storearea)
        :precondition (and
            (at ?h ?from)
            (clear ?to)
            (connected ?from ?to))
        :effect (and
            (not (at ?h ?from))
            (at ?h ?to)
            (not (clear ?to))
            (clear ?from))
    )

    (:action go-out
        :parameters (?h - hoist ?from - storearea ?to - transitarea)
        :precondition (and
            (at ?h ?from)
            (connected ?from ?to))
        :effect (and
            (not (at ?h ?from))
            (at ?h ?to)
            (clear ?from))
    )

    (:action go-in
        :parameters (?h - hoist ?from - transitarea ?to - storearea)
        :precondition (and
            (at ?h ?from)
            (connected ?from ?to)
            (clear ?to))
        :effect (and
            (not (at ?h ?from))
            (at ?h ?to)
            (not (clear ?to)))
    )
)
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
(define (problem storage-199)
	(:domain Storage-Propositional)
	(:objects
		depot48-1-1 depot48-1-2 depot49-1-1 depot49-1-2 depot50-1-1 container-0-0 container-1-0 container-2-0 - storearea
		hoist0 hoist1 - hoist
		crate0 crate1 crate2 - crate
		container0 container1 container2 - container
		depot48 depot49 depot50 - depot
		loadarea - transitarea
	)

	(:init
		(connected depot48-1-1 depot48-1-2)
		(connected depot48-1-2 depot48-1-1)
		(connected depot49-1-1 depot49-1-2)
		(connected depot49-1-2 depot49-1-1)
		(in depot48-1-1 depot48)
		(in depot48-1-2 depot48)
		(in depot49-1-1 depot49)
		(in depot49-1-2 depot49)
		(in depot50-1-1 depot50)
		(on crate0 container-0-0)
		(on crate1 container-1-0)
		(on crate2 container-2-0)
		(in crate0 container0)
		(in crate1 container1)
		(in crate2 container2)
		(in container-0-0 container0)
		(in container-1-0 container1)
		(in container-2-0 container2)
		(connected loadarea container-0-0)
		(connected container-0-0 loadarea)
		(connected loadarea container-1-0)
		(connected container-1-0 loadarea)
		(connected loadarea container-2-0)
		(connected container-2-0 loadarea)
		(connected depot48-1-1 loadarea)
		(connected loadarea depot48-1-1)
		(connected depot49-1-1 loadarea)
		(connected loadarea depot49-1-1)
		(connected depot50-1-1 loadarea)
		(connected loadarea depot50-1-1)
		(clear depot48-1-1)
		(clear depot49-1-1)
		(clear depot50-1-1)
		(at hoist0 depot48-1-2)
		(available hoist0)
		(at hoist1 depot49-1-2)
		(available hoist1)
	)

	(:goal
		(and
			(in crate0 depot48)
			(in crate1 depot48)
			(in crate2 depot49))
	)
)
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.PLAN_START.value}
(move hoist1 depot49-1-2 depot49-1-1)
(go-out hoist1 depot49-1-1 loadarea)
(lift hoist1 crate0 container-0-0 loadarea container0)
(drop hoist1 crate0 depot48-1-1 loadarea depot48)
(lift hoist1 crate2 container-2-0 loadarea container2)
(drop hoist1 crate2 depot49-1-1 loadarea depot49)
(lift hoist1 crate1 container-1-0 loadarea container1)
(lift hoist0 crate0 depot48-1-1 depot48-1-2 depot48)
(move hoist0 depot48-1-2 depot48-1-1)
(drop hoist0 crate0 depot48-1-2 depot48-1-1 depot48)
(go-out hoist0 depot48-1-1 loadarea)
(drop hoist1 crate1 depot48-1-1 loadarea depot48)
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}

Here is a checklist to help you with your task:

{config.TOKENS.CHECKLIST_START.value}
1) Do not provide any additional text or explanations outside the plan tags.
2) The plan must be in the same format as the examples above.
3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
{config.TOKENS.CHECKLIST_END.value}""")


from learning_to_plan.prompt_builder import base
from learning_to_plan import config
from learning_to_plan.data import task
from learning_to_plan.domain_translators import utils as domain_translator_utils

logger = config.get_logger(__name__)

class PDDLPromptBuilder(base.PromptBuilder):
    def __init__(self, **kwargs):
        super().__init__(prompt_type=config.PROMPT_TYPE.PDDL, **kwargs)

    def get_chat(self, t : task.Task, with_plan: bool = True, **kwargs) -> list[dict[str, str]]:
        content = PDDL_TEMPLATE_PROMPT.substitute(
            domain= t.domain,
            domain_description= t.read_domain(),
            instance_content= t.read_instance()
        )
        chat = [
            {"role": "system", "content": "You are an expert in AI Planning."},
            {"role": "user", "content": content}
        ]
        if with_plan:
            chat.append({"role": "assistant", "content": f"{config.TOKENS.PLAN_START.value}\n{t.pddl_plan}\n{config.TOKENS.PLAN_END.value}"})
        
        return chat
    
    def process_response(self, response: str) -> str:
        """
        Processes the response from the model.
        :param response: The response from the model.
        :return: The processed response.
        """
        if response == "":
            raise ValueError("The response is empty. Please provide a valid plan.")
        
        if config.TOKENS.PLAN_START.value not in response:
            raise ValueError(f"The response does not contain the plan start token '{config.TOKENS.PLAN_START.value}'. Please provide a valid plan.")
        
        if config.TOKENS.PLAN_END.value not in response:
            raise ValueError(f"The response does not contain the plan end token '{config.TOKENS.PLAN_END.value}'. Please provide a valid plan.")
        
        plan_start_index = response.index(config.TOKENS.PLAN_START.value)
        plan_end_index = response.index(config.TOKENS.PLAN_END.value)

        if plan_start_index > plan_end_index:
            raise ValueError("The plan start token is after the plan end token. Please provide a valid plan.")
        
        plan = response[plan_start_index + len(config.TOKENS.PLAN_START.value):plan_end_index].strip()

        if plan == "":
            raise ValueError("After removing the plan start and end tokens, the plan is empty. Please provide a valid plan.")
        
        return plan
        
