# Imports -----------------------------------------------------------------------------------------------------------
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing import List, Dict, Any, Optional
import inspect

# Library Imports ---------------------------------------------------------------------------------------------------
from miragram.src.base.base import MiraResponse
from miragram.src.call.call_base import LLM_Call


# Response Models ----------------------------------------------------------------------------------------------------
# Code llm decorator functions:
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

# Functionality List Decorator
# --------------------------------------------------------------------------------------------------------------------
functionality_requirements_prompt = """
    SYSTEM:
    You are an expert Software Architect who excels at planning and designing software systems.

    Examine the high-level goal for the software system, and generate a list of functionality requirements that will be needed to achieve that goal. 




"""


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


# Self Ask Decorator
# --------------------------------------------------------------------------------------------------------------------

self_ask_prompt = """
    Examples:
    {examples:lists}

    Query: {query}
"""
self_ask_dec = LLM_Call(
    prompt_template=self_ask_prompt,
    response_model=SelfAskCodeResponse,
    version="0001",
    category="self_ask_code",
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
finish_code_dec = LLM_Call(
    prompt_template=finish_code_prompt,
    response_model=CodeResponse,
    version="0001",
    category="finish_code",
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
gen_tests_dec = LLM_Call(
    prompt_template=gen_tests_prompt,
    response_model=TestLibrary,
    version="0001",
    category="gen_tests",
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
gen_test_library_dec = LLM_Call(
    prompt_template=gen_test_library_prompt,
    response_model=CodeResponse,
    version="0001",
    category="gen_test_library",
)
