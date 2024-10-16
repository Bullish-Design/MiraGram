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
from miragram.src.prompt.prompt_generator import (
    PromptGenerator,
    build_prompt_input_str,
    build_system_init_str,
    PromptExtraInput,
    SystemGoal,
    PromptInputGroup,
    SystemUserLink,
    code_input,
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


prompt = """You are an expert software architect who can write excellent software test suites.

        Examine the given python code and a list of test cases for that code, then - with the desired goal of that python code in mind - think about 
        how best to create a python test suite for that code and the given test cases. Then create and return that test suite, as well as an explanation
        of what that test suite does and how it works.

        Ensure all necessary test data is included in the test suite as variables so it can be used in the test suite. 

        The test suite should be a self contained file that is able to be run and return a report of all passing and failing tests. 

        *** IMPORTANT NOTE *** When importing the code to the test suite, you must use "from #IMPORT_LIB import..." as stand in text for the file import location. """


# Classes -------------------------------------------------------------------------------------------------------------

# Instances -----------------------------------------------------------------------------------------------------------


# Misc ----------------------------------------------------------------------------------------------------------------


# End -----------------------------------------------------------------------------------------------------------------
