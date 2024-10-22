# Imports -----------------------------------------------------------------------------------------------------------
from mirascope.core import prompt_template
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing import List, Dict, Any, Optional
import inspect

# Library Imports ---------------------------------------------------------------------------------------------------
from miragram.src.base.base import MiraResponse
from miragram.src.call.call_base import LLM_Call, MiraCall, LLM_Call_Options
from miragram.src.prompt.prompt_base import (
    PromptBase,
    PromptModel,
    IndividualPrompt,
    PromptInstance,
    UserPrompt,
    SystemPrompt,
    PromptGroup,
    PromptModifiers,
    PromptContainer,
)

# Variables ----------------------------------------------------------------------------------------------------------
user_prompt_list = []
system_prompt_list = []
PromptContainer = PromptContainer


# Response Models ----------------------------------------------------------------------------------------------------
# Code llm decorator functions:
class PydanticAttribute(MiraResponse):
    name: str = Field(description="The name of a Pydantic class attribute")
    attr_type: str = Field(description="The type of a pydantic attribute's value")
    description: str = Field(description="The description of a pydantic attribute.")


class PydanticResponse(MiraResponse):
    description: str = Field(
        description="Description of the Pydantic model. Will be used as the class docstring."
    )
    class_name: str = Field(
        description="Class name of the Pydantic model. Should follow good naming practices."
    )
    inherits: str = Field(
        description="The inherited class for the Pydantic model. Is generally 'BaseModel', which represents the generic Pydantic base model."
    )
    # attributes: Dict[str, str] = Field(
    #    description="Attributes of the Pydantic model, returned as a dictionary with the attribute name as the key and the attribute type as the value. If the attribute has a 'Field(description=...', that should be included in the value of the
    # )
    attributes: List[PydanticAttribute]


class SystemRequirement(MiraResponse):
    requirement_name: str = Field(
        description="The requirement name, for a requirement of a software system."
    )
    requirement_description: str = Field(
        description="The description of a software requirement for a software system."
    )


class SystemRequirementsList(MiraResponse):
    requirements: List[SystemRequirement]


class RequirementMilestone(MiraResponse):
    description: str = Field(description="A brief description of the milestone")
    why_milestone: str = Field(
        description="A brief description of how this milestone will help reach the ultimate goal of the system requirement."
    )
    mvp_description: str = Field(
        description="A brief description of the minimum viable product that will determine when this milestone is met."
    )


class RequirementMilestoneList(MiraResponse):
    milestone_list: List[RequirementMilestone]


class ClassIdea(MiraResponse):
    description: str = Field(
        description="A description of a Pydantic class that should be created in order to meet the needs of a software system."
    )


class ClassIdeas(MiraResponse):
    idea_list: List[ClassIdea] = Field(
        description="A list of Pydantic class model ideas."
    )


class ParserInput(MiraResponse):
    description: str = Field(description="Description of the input to be parsed")
    input: str = Field(description="The input to be parsed")


class ParserOutput(MiraResponse):
    description: str = Field(
        description="Description of the expected output after parsing"
    )
    output: str = Field(description="The expected output after parsing.")


class ParsyFunction(MiraResponse):
    description: str = Field(
        description="Description of the Parsy parsing function - Will be used as the function Docstring."
    )
    parse_function_code: str = Field(
        description="The code for the Parsy parsing function. Uses the '@generate' decorator to create a Parsy parser function."
    )


class ProjectDirectory(MiraResponse):
    parent_directory: str = Field(
        description="The name of the parent directory this directory exists within."
    )
    directory_readme: str = Field(
        description="The contents of the readme file that exists in every folder within the project. Describes the purpose and functionality of the code contained within."
    )


class MarkdownSection(MiraResponse):
    title: str = Field(description="Title of a markdown section")
    content: str = Field(description="Content of a markdown section")


class MarkdownResponse(MiraResponse):
    sections: List[MarkdownSection] = Field(
        description="List of markdown sections that make up the response."
    )


class FewShot(MiraResponse):
    question: str
    answer: str


class CodeBlock(MiraResponse):
    code: str = Field(
        description="Detailed python code that solves the user request in the form of a single formatted code block."
    )


class CodeResponse(MiraResponse):
    explanation: str = Field(
        description="All explanation or preamble necessary to understand the code goes here. There should be no explanation in the code itself aside from comments."
    )
    code: CodeBlock


class TestData(MiraResponse):
    input: str = Field(description="Test Input Data")
    output: str = Field(
        description="The desired output data after running the test case"
    )


class TestCase(MiraResponse):
    description: str = Field(
        description="A brief description detailing why the test case is needed."
    )
    test_data: list[TestData]
    test_case: CodeBlock


class TestLibrary(MiraResponse):
    test_cases: list[TestCase]


class SelfAskCodeResponse(MiraResponse):
    self_ask: list[FewShot]
    response: CodeResponse


# Define few-shot examples
# --------------------------------------------------------------------------------------------------------------------

few_shot_code_examples = [
    FewShot(
        question="Please create a python function that adds two numbers",
        answer=inspect.cleandoc(
            """
            def add(a, b):
                '''Add two numbers and return the result.'''
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise ValueError("Both inputs must be numbers.")
                return a + b            
            """
        ),
    ),
    FewShot(
        question="Please create a python function that subtracts one number from another",
        answer=inspect.cleandoc(
            """
            def subtract(a, b):
                '''Subtract second number from first and return result.'''
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise ValueError("Both inputs must be numbers.")
                return a - b
            """
        ),
    ),
    # ... (add more examples here)
]

parsy_few_shot_examples = [
    FewShot(
        question="Please create a Parsy parser that can parse a date in the format YYYY-MM-DD",
        answer=inspect.cleandoc(
            """
@generate
def date():
    ''''
    Parse a date in the format YYYY-MM-DD
    ''''
    year = yield regex("[0-9]{4}").map(int)
    yield string("-")
    month = yield regex("[0-9]{2}").map(int)
    yield string("-")
    day = yield regex("[0-9]{2}").map(int)

    return date(year, month, day)

"""
        ),
    )
]
# Functionality List Decorator
# --------------------------------------------------------------------------------------------------------------------


test_markdown_section_prompt_system = """You are an expert Software Architect who excels at planning and designing software systems. 

    Examine the functionality request below and think about how best to integrate the given request into a software system. Then write a detailed response that outlines how the request should be integrated into a software system. 

    The response should consist of a Markdown section with a header that briefly describes the section, followed by a detailed text-only explanation of how best to implement the request in a software system."""

test_markdown_section_prompt_user = """Here is my overall high-level goal for the software system: {goal}
    Here is the specific functionality I want to achieve: {request}
    Explain how best to integrate this functionality into a software system."""

test_markdown_section_prompt = PromptBase(
    user_prompt=test_markdown_section_prompt_user,
    system_prompt=test_markdown_section_prompt_system,
)

markdown_section_user = UserPrompt(
    name="markdown_user",
    prompt=PromptInstance(prompt=test_markdown_section_prompt_user),
)
markdown_section_system = SystemPrompt(
    name="markdown_system",
    prompt=PromptInstance(prompt=test_markdown_section_prompt_system),
)

user_prompt_list.append(markdown_section_user)
system_prompt_list.append(markdown_section_system)

# Pydantic class generator
# --------------------------------------------------------------------------------------------------------------------
functionality_requirements_prompt = """
    You are an expert Software Architect who excels at planning and designing software systems.

    Examine the high-level goal for the software system, and then with that in mind, respond to the user request.
"""

mvp_test_class = """Below is a brief description of a minimum viable requirement for a software project. The ultimate goal of the software project is {goal}. Please create a test function that, when passed, represents achieving full functionality of the MVP.

"""

user_create_requirements = """Here's my goal for the software system: {goal}.
Please generate a list of functionality requirements that will be needed to achieve that goal. Requirements should be in the form of a Name, and then a brief description. Only consider technical requirements - do not include any requirements related to documentation, stakeholder/user feedback, etc.
"""

user_create_pydantic = """Here's my goal for the software system: {goal}.
Please generate a list of descriptions of pydantic class models that would help me to achieve that goal."""

user_create_milestones = """Here's my goal for the software system: {goal}.
Here's a system requirement that is required to meet that ultimate goal: {requirement}
Please generate a list of milestones that, if achieved, would ensure that I would achieve that system requirement."""

miracall_milestones = MiraCall(
    system_prompt=functionality_requirements_prompt,
    prompt_template=user_create_milestones,
    response_model=RequirementMilestoneList,
    version="0001",
    category="project_requirements",
)

miracall_requirements = MiraCall(
    system_prompt=functionality_requirements_prompt,
    prompt_template=user_create_requirements,
    response_model=SystemRequirementsList,
    version="0001",
    category="project_requirements",
)

pydantic_ideas = MiraCall(
    system_prompt=functionality_requirements_prompt,
    prompt_template=user_create_requirements,
    response_model=ClassIdeas,
    version="0001",
    category="project_requirements",
)

pydantic_system_prompt = """You are an expert software engineer who can write good clean code and is great at creating Pydantic models. Your models contain all necessary attributes, and have excellent class and attribute names that perfectly describe their purpose. """

pydantic_user_prompt = """Here is the description of the input data that needs to be stored in the pydantic model: {description}
Please create a Pydantic model that represents the given input data."""

miracall_pydantic_request = MiraCall(
    system_prompt=pydantic_system_prompt,
    prompt_template=pydantic_user_prompt,
    response_model=PydanticResponse,
    version="0001",
    category="pydantic_model_request",
)

# Parsy Parser Generator:
# --------------------------------------------------------------------------------------------------------------------
parsy_system_prompt = """You are an expert software engineer who can write good clean code and excels at creating parsers using the Parsy python library. Your parsers are clean, efficient, and easy to understand. You believe it is more important for parsers to be easily understandable to a layperson, so you always write your parsers with that in mind. You like to write out each step for the parsing function on a separate line, using underscores to denote which pieces of parsed data are not being returned.

Examine the parser request below and think about how best to implement the given parser request. Then create a Parsy parser function that can parse the given input data and return the expected output data. Ensure the parser function uses the '@generate' decorator method of creating a Parsy parser."""

parsy_user_request = """Here is the input data that needs to be parsed: {input}
Here is a description of the expected output data after parsing: {output}
Please create a Parsy parser function that can parse the given input data and return the expected output data."""

miracall_parsy_request = MiraCall(
    system_prompt=parsy_system_prompt,
    prompt_template=parsy_user_request,
    response_model=ParsyFunction,
    version="0001",
    category="parsy_parser",
)

# Markdown Section Decorator
# --------------------------------------------------------------------------------------------------------------------
markdown_section_prompt = """
    SYSTEM: 
    You are an expert Software Architect who excels at planning and designing software systems. 

    Examine the functionality request below and think about how best to integrate the given request into a software system. Then write a detailed response that outlines how the request should be integrated into a software system. 

    The response should consist of a Markdown section with a header that briefly describes the section, followed by a detailed text-only explanation of how best to implement the request in a software system. 

    USER:
    Here is my overall high-level goal for the software system: {goal}
    Here is the specific functionality I want to achieve: {request}
    Explain how best to integrate this functionality into a software system.

"""

functionality_explanation_dec = LLM_Call(
    prompt_template=markdown_section_prompt,
    response_model=MarkdownSection,
    version="0001",
    category="functionality_explanation",
)

miracall_functionality_explanation_dec = MiraCall(
    system_prompt=None,
    prompt_template=markdown_section_prompt,
    response_model=MarkdownSection,
    version="0001",
    category="functionality_explanation",
)

# Self Ask Decorator
# --------------------------------------------------------------------------------------------------------------------

self_ask_prompt = """
    Examples:
    {examples:lists}

    Query: {query}
"""
self_ask_user = UserPrompt(
    name="self_ask_user", prompt=PromptInstance(prompt=self_ask_prompt)
)

user_prompt_list.append(self_ask_user)

self_ask_dec = LLM_Call(
    prompt_template=self_ask_prompt,
    response_model=SelfAskCodeResponse,
    version="0001",
    category="self_ask_code",
)

miracall_self_ask_dec = MiraCall(
    system_prompt=None,
    prompt_template=self_ask_prompt,
    response_model=SelfAskCodeResponse,
    json_response=True,
)

# Finish Code Decorator
# --------------------------------------------------------------------------------------------------------------------
finish_code_prompt = """
        SYSTEM:
        You are an expert software engineer who can write good clean code and solve
        complex problems.

        Examine the given python code and - with the desired goal of that python code in mind - think about 
        how best to take the code from its current state to fully finalized and running code. Then rewrite and return the 
        given code in its final fully functional state as a single code block, as well as an explanation of the code in its final state.


        USER:
        Here is my goal for the following python code: {goal}
        Here is the code I have created so far: {code}
        Please make any additions or updates that are neccessary so the code accomplishes my goal, and return the complete code.
        """
finish_code_user = UserPrompt(
    name="finish_code_user",
    prompt=PromptInstance(
        prompt="""Here is my goal for the following python code: {goal}
        Here is the code I have created so far: {code}
        Please make any additions or updates that are neccessary so the code accomplishes my goal, and return the complete code."""
    ),
)

finish_code_system = SystemPrompt(
    name="finish_code_system",
    prompt=PromptInstance(
        prompt="""You are an expert software engineer who can write good clean code and solve
        complex problems.

        Examine the given python code and - with the desired goal of that python code in mind - think about 
        how best to take the code from its current state to fully finalized and running code. Then rewrite and return the 
        given code in its final fully functional state as a single code block, as well as an explanation of the code in its final state."""
    ),
)

user_prompt_list.append(finish_code_user)
system_prompt_list.append(finish_code_system)

finish_code_dec = LLM_Call(
    prompt_template=finish_code_prompt,
    response_model=CodeResponse,
    version="0001",
    category="finish_code",
)

miracall_finish_code_dec = MiraCall(
    system_prompt=None,
    prompt_template=finish_code_prompt,
    response_model=CodeResponse,
    json_response=True,
)

# Generate Tests
# --------------------------------------------------------------------------------------------------------------------

gen_tests_prompt = """
        SYSTEM:
        You are an expert software reliability engineer who can write excellent software test cases and is great at 
        choosing test data.

        Examine the given python code and - with the desired goal of that python code in mind - think about 
        how best to test all aspects of the code. Then create and return a Test Library. The Test Library consists of a 
        list of test cases, where each test case consists of:
            description: A brief description detailing why the test case is needed.
            test_data: A list of input data, as well as the corresponding desired output data.
            test_case: The code block that uses the pytest library to test some aspect of the python code.

        All test cases should use the pytest library.


        USER:
        Here is my goal for the following python code: {goal}
        Here is my code: {code}
        Please give me a list of test cases that I can use to ensure my code doesnt fail in production.
        """
test_gen_user = UserPrompt(
    name="test_gen_user",
    prompt=PromptInstance(
        prompt="""Here is my goal for the following python code: {goal}
        Here is my code: {code}
        Please give me a list of test cases that I can use to ensure my code doesnt fail in production."""
    ),
)
test_gen_system = SystemPrompt(
    name="test_gen_system",
    prompt=PromptInstance(
        prompt=""" You are an expert software reliability engineer who can write excellent software test cases and is great at 
        choosing test data.

        Examine the given python code and - with the desired goal of that python code in mind - think about 
        how best to test all aspects of the code. Then create and return a Test Library. The Test Library consists of a 
        list of test cases, where each test case consists of:
            description: A brief description detailing why the test case is needed.
            test_data: A list of input data, as well as the corresponding desired output data.
            test_case: The code block that uses the pytest library to test some aspect of the python code.

        All test cases should use the pytest library."""
    ),
)

user_prompt_list.append(test_gen_user)
system_prompt_list.append(test_gen_system)

gen_tests_dec = LLM_Call(
    prompt_template=gen_tests_prompt,
    response_model=TestLibrary,
    version="0001",
    category="gen_tests",
)

miracall_gen_tests_dec = MiraCall(
    system_prompt=None,
    prompt_template=gen_tests_prompt,
    response_model=TestLibrary,
    json_response=True,
)
# Generate Test Library
# --------------------------------------------------------------------------------------------------------------------

gen_test_library_prompt = """
        SYSTEM:
        You are an expert software architect who can write excellent software test suites.

        Examine the given python code and a list of test cases for that code, then - with the desired goal of that python code in mind - think about 
        how best to create a python test suite for that code and the given test cases. Then create and return that test suite, as well as an explanation
        of what that test suite does and how it works.

        Ensure all necessary test data is included in the test suite as variables so it can be used in the test suite. 

        The test suite should be a self contained file that is able to be run and return a report of all passing and failing tests. 

        *** IMPORTANT NOTE *** When importing the code to the test suite, you must use "from #IMPORT_LIB import..." as stand in text for the file import location.


        USER:
        Here is my goal for the following python code: {goal}
        Here is my code: {code}
        Here is a list of test cases: {test_cases:list}
        Here is a list of test data to be used in testing: {test_data:list} 
        Please give me a full pytest test suite that provides front to back test coverage for my given code.
        """
test_library_user = UserPrompt(
    name="test_library_user",
    prompt=PromptInstance(
        prompt="""Here is my goal for the following python code: {goal}
        Here is my code: {code}
        Here is a list of test cases: {test_cases:list}
        Here is a list of test data to be used in testing: {test_data:list} 
        Please give me a full pytest test suite that provides front to back test coverage for my given code."""
    ),
)

test_library_system = SystemPrompt(
    name="test_library_system",
    prompt=PromptInstance(
        prompt="""You are an expert software architect who can write excellent software test suites.

        Examine the given python code and a list of test cases for that code, then - with the desired goal of that python code in mind - think about 
        how best to create a python test suite for that code and the given test cases. Then create and return that test suite, as well as an explanation
        of what that test suite does and how it works.

        Ensure all necessary test data is included in the test suite as variables so it can be used in the test suite. 

        The test suite should be a self contained file that is able to be run and return a report of all passing and failing tests. 

        *** IMPORTANT NOTE *** When importing the code to the test suite, you must use "from #IMPORT_LIB import..." as stand in text for the file import location. """
    ),
)

user_prompt_list.append(test_library_user)
system_prompt_list.append(test_library_system)

gen_test_library_dec = LLM_Call(
    prompt_template=gen_test_library_prompt,
    response_model=CodeResponse,
    version="0001",
    category="gen_test_library",
)

miracall_gen_test_library_dec = MiraCall(
    system_prompt=None,
    prompt_template=gen_test_library_prompt,
    response_model=CodeResponse,
    json_response=True,
)

user_prompts = PromptGroup(prompts=user_prompt_list)
system_prompts = PromptGroup(prompts=system_prompt_list)

architect_prompts = PromptModel(
    user_prompts=user_prompts, system_prompts=system_prompts
)
