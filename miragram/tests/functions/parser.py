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

# Typer Script ------------------------------------------------------------------------------------------------------
import typer
from pathlib import Path
from typing import Optional
import sys

'''
# Create a Typer app instance
app = typer.Typer(help="Example script that can be called from pyproject.toml")


@app.command()
def _main(
    input_file: Path = typer.Argument(
        ...,  # ... means required
        help="Path to input file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        "./output",
        "--output-dir",
        "-o",
        help="Directory for output files",
        dir_okay=True,
        file_okay=False,
        resolve_path=True,
        create_dir=True,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> int:
    """
    Process input file and save results to output directory.
    """
    try:
        if verbose:
            typer.echo(f"Processing {input_file}")
            typer.echo(f"Output directory: {output_dir}")

        # Your script logic here

        return 0

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        return 1
'''


def askGPT(request: str):
    """
    Entry point function for pyproject.toml script definition.
    """
    print(f"\n\nStarting up script....\n\n")
    return


# Variables ----------------------------------------------------------------------------------------------------------
user_prompt_list = []
system_prompt_list = []
PromptContainer = PromptContainer


# Response Models ----------------------------------------------------------------------------------------------------
# Code llm decorator functions:


# This class describes the shape of an attribute for a Pydantic class
class PydanticAttribute(MiraResponse):
    name: str = Field(description="The name of a Pydantic class attribute")
    attr_type: str = Field(
        description="The type of a pydantic attribute's value. Could be a standard python type, or any of the created Pydantic models generated in the user request."
    )
    description: str = Field(description="The description of a pydantic attribute.")


# This class describes the shape of an input argument for a python function
class FunctionInput(MiraResponse):
    name: str = Field(description="The name of the input argument for a function.")
    arg_type: str = Field(description="The type for an input argument of a function.")


# This class describes the shape of a function for a Pydantic class.
class PydanticFunction(MiraResponse):
    name: str = Field(description="The name of a function in a Pydantic class.")
    input_args: List[FunctionInput]  # = Field(description="")
    description: str = Field(
        description="The function description of a function in a pydantic class. This will be used as the docstring, so be sure to include information from the input_args and describe what the function does to them, before finishing with the output description and it's type."
    )
    output_type: str = Field(
        description="The type(s) for the output(s) from the function."
    )


# This class describes the shape of a Pydantic class - the name, description, inheritance, attributes, and functions.
class PydanticResponse(MiraResponse):
    description: str = Field(
        description="Description of the Pydantic model. Will be used as the class docstring."
    )
    class_name: str = Field(
        description="Class name of the Pydantic model. Should follow good naming practices."
    )
    inherits: str = Field(
        description="The inherited class for the Pydantic model. Is generally 'BaseModel', which represents the generic Pydantic base model, but could also be one of the other Pydantic classes generated from the user request."
    )
    attributes: List[PydanticAttribute]
    functions: List[PydanticFunction] = Field(
        description="A list of functions contained within the Pydantic class, if any."
    )


# This class describes the shape of a system requirement for a software project
class SystemRequirement(MiraResponse):
    requirement_name: str = Field(
        description="The requirement name, for a requirement of a software system."
    )
    requirement_description: str = Field(
        description="The description of a software requirement for a software system."
    )


# This class describes the shape of a core functionality for a software project
class CoreFunctionality(MiraResponse):
    functionality_name: str = Field(
        description="The name of the core functionality category"
    )
    functionality_description: str = Field(
        description="A brief description of the core functionality comprised within the category."
    )


# This class describes the shape of a list of functionality for a software project
class FunctionalityList(MiraResponse):
    functionality_list: List[CoreFunctionality]


# This class describes the shape of a list of requirements for a software project
class SystemRequirementsList(MiraResponse):
    requirements: List[SystemRequirement]


# This class describes the shape of a requirement milestone for a core functionality of a software project.
class RequirementMilestone(MiraResponse):
    description: str = Field(description="A brief description of the milestone")
    why_milestone: str = Field(
        description="A brief description of how this milestone will help reach the ultimate goal of the system requirement."
    )
    mvp_description: str = Field(
        description="A brief description of the minimum viable product that will determine when this milestone is met."
    )


# This class describes the shape of a list of requirement milestones for a software project, each one getting successively more feature filled and refined.
class RequirementMilestoneList(MiraResponse):
    milestone_list: List[RequirementMilestone]


# This class describes the shape of a
class ClassIdea(MiraResponse):
    description: str = Field(
        description="A description of a Pydantic class that should be created in order to meet the needs of a software system."
    )


# This class describes the shape of a
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


class ProjectFile(MiraResponse):
    name: str = Field(
        description="The name of the file, including the filetype suffix."
    )
    # file_type: str = Field(description="The file type. Examples would be 'Text, Python, Markdown')
    description: str = Field(
        description="A brief description of the file contents, and it's purpose within the overall project."
    )


class ProjectDirectory(MiraResponse):
    name: str = Field(description="The name of the directory.")
    directory_readme: str = Field(
        description="The contents of the readme file that exists in every folder within the project. Describes the purpose and functionality of the files and subfolders contained within."
    )
    sub_folders: List["ProjectDirectory"] = Field(
        description="A list of folders underneath the current folder"
    )
    files: List[ProjectFile]


class FewShot(MiraResponse):
    question: str
    answer: str


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


# System requirements Generation
# --------------------------------------------------------------------------------------------------------------------
functionality_requirements_prompt = """
    You are an expert Software Architect who excels at planning and designing software systems.

    Examine the high-level goal for the software system, and then with that in mind, respond to the user request.
"""

mvp_test_class = """Below is a brief description of a minimum viable requirement for a software project. The ultimate goal of the software project is {goal}. Please create a test function that, when passed, represents achieving full functionality of the MVP.

"""

core_functionality_requirements = """Here is my overall goal for the software system: {goal}.

Given the following list of requirements for the software system, distill the functionality down to the core functionality 'modules' the software system will contain. Group common functionality by what would commonly exist together in production software systems. For example: If the system has requirements for file creation, file reading, file writing, and file renaming, these would generally be grouped into a functional module like 'FileIO'.

Here is the list of requirements for the software system: {requirements}

Please provide the grouped/condensed list of core functionality for the software system.
"""

user_create_requirements = """Here's my goal for the software system: {goal}.
Please generate a list of functionality requirements that will be needed to achieve that goal. Requirements should be in the form of a Name, and then a brief description. Only consider technical requirements - do not include any requirements related to documentation, stakeholder/user feedback, etc.
"""

user_create_milestones = """Here's my goal for the software system: {goal}.
Here's a core functionality that is required to meet that ultimate goal: {functionality}
Please generate a list of milestones that, if achieved, would ensure that I would fully achieve that system requirement."""

miracall_condensed_functionality = MiraCall(
    system_prompt=functionality_requirements_prompt,
    prompt_template=core_functionality_requirements,
    response_model=FunctionalityList,
    version="0001",
    category="project_requirements",
)

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

# Pydantic Classes:
user_create_pydantic = """Here's my goal for the software system: {goal}.
Here is a list of core functionality groups that the software system will need to be able to handle: {functionality_groups}
Please generate a list of descriptions of pydantic class models that would help me to achieve that goal. """


pydantic_system_prompt = """You are an expert software engineer who can write good clean code and is great at creating Pydantic models. Your models contain all necessary attributes, and have excellent class and attribute names that perfectly describe their purpose. You excel at creating abstraction layers to enable software developers to quickly accomplish their goals when using the Pydantic models you've created. 

Consider the overall goal of the software system described below, as well as the functional categories the software system is grouped into. Then, with all that in mind, think carefully of how best to structure a hierarchy of Pydantic models, and then respond to the user's request. Ensure that any pydantic classes created integrate well within the system's structure. Abstraction is incredibly important to ensure rapid development, so consider class inheritance before you create any classes."""

pydantic_user_prompt = """Here's my overall goal for the software system: {goal}.

Here is the description the specific functionality that this class will need to provide: {description}
Please create a Pydantic model for the described class. """

pydantic_ideas = MiraCall(
    system_prompt=pydantic_system_prompt,
    prompt_template=user_create_pydantic,
    response_model=ClassIdeas,
    version="0001",
    category="project_requirements",
)

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


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


# ====================================================================================================


class MarkdownSection(MiraResponse):
    title: str = Field(description="Title of a markdown section")
    content: str = Field(description="Content of a markdown section")


class MarkdownResponse(MiraResponse):
    sections: List[MarkdownSection] = Field(
        description="List of markdown sections that make up the response."
    )


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
