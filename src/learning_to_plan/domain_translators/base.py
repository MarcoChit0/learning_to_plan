from __future__ import annotations
import abc
import re
from learning_to_plan import config
from learning_to_plan.data import task
logger = config.get_logger(__name__)

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
        pattern = r"\({}\s+(.*?)(?=\s+\(:|\s*\)\s*$)".format(re.escape(pattern=section_name))
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