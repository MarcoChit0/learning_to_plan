from learning_to_plan.domain_translators.base import DomainTranslator
import re

class BlocksworldTranslator(DomainTranslator):
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
                    # Correct case for one-block actions: block2_id is None, nothing else needed

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