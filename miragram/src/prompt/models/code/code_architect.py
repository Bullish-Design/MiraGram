# Imports -------------------------------------------------------------------------------------------------------------
from pydantic import BaseModel, computed_field, PrivateAttr
from typing import List, Optional, Dict

# Local imports -------------------------------------------------------------------------------------------------------
from miragram.src.prompt.prompt_base import (
    PromptInstance,
    IndividualPrompt,
    PromptType,
    UserPrompt,
    SystemPrompt,
    UserModifierPrompt,
    SystemModifierPrompt,
    PromptContainer,
    PromptModifiers,
    PromptGroup,
    PromptBase,
    PromptModel,
)

# Logging -------------------------------------------------------------------------------------------------------------
from miragram.log.logger import get_logger

logger = get_logger("CodeArchitect_Prompts")

# Prompts -------------------------------------------------------------------------------------------------------------
code_goal = """Here is my goal for the following python code: {goal}"""
test_cases = """ Here is my code: {code}
        Here is a list of test cases: {test_cases:list}
        Here is a list of test data to be used in testing: {test_data:list} """


# Functions -----------------------------------------------------------------------------------------------------------
def build_prompt_input_str(
    input_name: str,
    input_use: Optional[str] = None,
    multiple: bool = False,
    use_modifier: Optional[str] = None,
) -> str:
    return_str = f"Here is the "
    input_variable = f"{{{input_name}}}"
    if multiple:
        return_str += "list of "
        input_variable = f"{{{input_name}:list}}"

    if input_use and not use_modifier:
        return_str += f"to be used for {input_use}"
    elif input_use and use_modifier:
        return_str += f"for {use_modifier} {input_use}"
    elif not input_use and use_modifier:
        return_str += f"for {use_modifier}"
    else:
        return_str += ""
    return_str += f": {input_variable}"
    return return_str


prompt = """You are an expert software architect who can write excellent software test suites.

        Examine the given python code and a list of test cases for that code, then - with the desired goal of that python code in mind - think about 
        how best to create a python test suite for that code and the given test cases. Then create and return that test suite, as well as an explanation
        of what that test suite does and how it works.

        Ensure all necessary test data is included in the test suite as variables so it can be used in the test suite. 

        The test suite should be a self contained file that is able to be run and return a report of all passing and failing tests. 

        *** IMPORTANT NOTE *** When importing the code to the test suite, you must use "from #IMPORT_LIB import..." as stand in text for the file import location. """


def build_system_init_str(system_name: str, system_goal: str) -> str:
    pass


# Classes -------------------------------------------------------------------------------------------------------------
class PromptExtraInput(BaseModel):
    input_name: str
    input_use: Optional[str] = None
    multiple: bool = False
    use_modifier: Optional[str] = None

    @computed_field
    def prompt(self) -> str:
        return build_prompt_input_str(
            self.input_name, self.input_use, self.multiple, self.use_modifier
        )

    def __call__(self) -> str:
        return self.prompt


class PromptInputGroup(BaseModel):
    input_group: List[PromptExtraInput]
    _group_dict: Dict[str, PromptExtraInput] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._create_group_dict()

    def _create_group_dict(self):
        group_dict = {group.input_name: group.prompt for group in self.input_group}
        self.__dict__["_group_dict"] = group_dict

    def __getattr__(self, name):
        group_dict = self._group_dict
        if name in group_dict:
            group_instance = group_dict[name]
            return group_instance
        raise AttributeError(
            f"\n\n'{self.__class__.__name__}' object has no attribute '{name}'\n"
        )


# Instances -----------------------------------------------------------------------------------------------------------
goal = PromptExtraInput(input_name="goal", use_modifier="the following python code")
code = PromptExtraInput(input_name="code")
test_cases = PromptExtraInput(input_name="test cases", multiple=True)
test_data = PromptExtraInput(
    input_name="test data", multiple=True, use_modifier="use in testing"
)
code_input = PromptInputGroup(input_group=[goal, code, test_cases, test_data])


# TODO: Is there a way to easily bundle up everything for easy export?


# Misc ----------------------------------------------------------------------------------------------------------------


# End -----------------------------------------------------------------------------------------------------------------
