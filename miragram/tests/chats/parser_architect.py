# Imports
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, ValidationError
from sqlalchemy.engine import result
from sqlmodel import SQLModel, Relationship, Field as SQLField
from typing import List, Dict, Any, Optional, cast, Annotated
from typing_extensions import TypedDict
import os
import json
from time import time
from datetime import datetime
from rich.console import Console
from rich.tree import Tree
from rich.syntax import Syntax
from rich import print as rprint

from openai import OpenAI
from mirascope.core import openai, prompt_template

# Local Imports
from miragram.src.base.base import (
    # IntermediateResponse,
    # ArchitectAskResult,
    # ArchitectAskStep,
    MiraResponse,
)
from miragram.src.call.call_base import (
    IntermediateResponse,
    get_single_instance_from_db,
    MiraCall,
    LLM_Call_Options,
)
from miragram.tests.functions.parser import (
    FewShot,
    TestLibrary,
    TestData,
    few_shot_code_examples,
    parsy_few_shot_examples,
    ParsyFunction,
    ParserInput,
    ParserOutput,
    miracall_parsy_request,
    self_ask_dec,
    finish_code_dec,
    gen_tests_dec,
    gen_test_library_dec,
    miracall_self_ask_dec,
    miracall_finish_code_dec,
    miracall_gen_tests_dec,
    miracall_gen_test_library_dec,
    # test_markdown_section_prompt,
    architect_prompts,
    user_prompts,
    system_prompts,
    miracall_pydantic_request,
    PydanticResponse,
    miracall_requirements,
    pydantic_ideas,
    SystemRequirementsList,
    ClassIdeas,
    miracall_milestones,
    RequirementMilestoneList,
    SystemRequirement,
    # PromptContainer,
    miracall_condensed_functionality,
    FunctionalityList,
    CoreFunctionality,
)
from miragram.src.prompt.prompt_base import (
    PromptContainer,
)

from miragram.tests.models.parser_models import GeneratedParser


# Configure logging
from miragram.src.base.config import (
    code_output_dir,
    db_url,
    local_model_url,
    local_model_name,
)

from miragram.log.logger import get_logger

logger = get_logger("SoftwareArchitect_Logging")

client = OpenAI(
    api_key="ollama",
    base_url=local_model_url,
)

# Constants ---------------------------------------------------------------------------------------------------------
code_request = """
    Please create a python script that takes a folder directory as argument. Inside that directory is 3 sub directories: "code", "data", and "tests". The python script should run all pytest functions inside the "tests" directory by running the code inside the "code" directory against all data in the "data" directory. 
    """
code_output_dir = code_output_dir

# Parsy Requests ----------------------------------------------------------------------------------------------------
input = "Please create a Parsy parser that can parse a date in the format YYYY-MM-DD"

input_req_1 = ParserInput(
    description="Please create a Parsy parser that can parse a date in the format YYYY-MM-DD",
    input="1992-12-31",
)
expected_output_1 = ParserOutput(
    description="The parser should return a datetime Date object with the corresponding values",
    output="datetime.date(1992, 12, 31)",
)


def parse_output_code(output_code: str):
    code_lines = output_code.split("\n")
    code_str = ""
    for line in code_lines:
        code_str += f"{line}\n"
    return code_str


def print_pydantic_model(model: PydanticResponse):
    print(f"\nclass {model.class_name}({model.inherits}):")
    print(f"""    '''
    {model.description}
    '''""")
    for attr in model.attributes:
        print(f"    {attr.name}: {attr.attr_type}  # {attr.description}")

    print(f"\n")
    for func in model.functions:
        args = ""
        for arg in func.input_args:
            args += f"{arg.name}: {arg.arg_type}, "
        if len(args) > 0:
            args = args[:-2]
        # args = f"{arg.name}: {arg.arg_type},"
        #    for arg in func.input_args
        #    if len(func.input_args) > 0

        print(f"    def {func.name}({args}) -> {func.output_type}: ...")
        print(f"""        '''
        {func.description}
        '''
        """)


def print_breakline():
    print(
        f"\n---------------------------------------------------------------------------------------------------\n"
    )


def print_milestone_requirements(req_list: RequirementMilestoneList):
    # print_breakline()
    # print(f"\n\n{req_list}\n\n\n")
    # print_breakline()
    for req in req_list.milestone_list:
        print(f"   Description: {req.description}")
        print(f"          Why?: {req.why_milestone}")
        print(f"           MVP: {req.mvp_description}\n")
    # print_breakline()


# Quick classes -----------------------------------------------------------------------------------------------------


# Parsy Functions ---------------------------------------------------------------------------------------------------
@miracall_condensed_functionality
def condense_functionality(
    goal: str, requirements: SystemRequirementsList
) -> FunctionalityList: ...


@miracall_parsy_request
def parsy_request(input: ParserInput, output: ParserOutput) -> ParsyFunction: ...


@miracall_pydantic_request
def pydantic_request(
    goal: str, functionality_groups: FunctionalityList, description: str
) -> PydanticResponse: ...


@miracall_requirements
def get_requirements(goal: str) -> SystemRequirementsList: ...


@pydantic_ideas
def get_class_ideas(
    goal: str, functionality_groups: FunctionalityList
) -> ClassIdeas: ...


@miracall_milestones
def create_milestones(
    goal: str, functionality: CoreFunctionality
) -> RequirementMilestoneList: ...


class Book(MiraResponse):
    """The JSON format for a book"""

    title: str = Field(description="The title of the book")
    author: str = Field(description="The author of the book")
    genre: str = Field(description="The genre of the book")
    summary: str = Field(description="A brief summary of the book")


class BookList(MiraResponse):
    """A list of books"""

    books: List[Book] = Field(description="A list of books")


def parse_json(response: openai.OpenAICallResponse):
    print(f"\n\nParsing Response...\n\n\n{response.response}\n\n\n")
    print(f"\nParsed Response: \n\n{response}\n\nDone with parsing")
    return response.content


@openai.call(local_model_name, client=client, json_mode=True, response_model=Book)
@prompt_template(
    """
    Please recommend a book in the {genre} genre.
    DO NOT pick a book from the following list: {current_list:list}
    Use the following exmaple as a response template for the response:
    {example}
    """
)
def test_pydantic_request(genre: str, current_list: list[str], example) -> str:
    ...
    # return f"Recommend a book in the {genre} genre."


@openai.call(local_model_name, client=client, json_mode=True, response_model=BookList)
@prompt_template(
    """
    Please recommend {num} unique books in the {genre} genre.
    Use the following exmaple as a reponse template for the response: 
    {example}
    DO NOT REPEAT BOOKS IN THE LIST. DO NOT ALLOW EXAMPLE BOOKS TO INFLUENCE RESULTS. DO NOT GIVE BACK THE EXAMPLE LIST.
    """
)
def test_book_list(num: str, genre: str, example) -> str:
    ...
    # return f"Recommend a book in the {genre} genre."


@openai.call(local_model_name, client=client)
@prompt_template(
    """You are a skilled software engineer. Please create a python script according to the following request:
    {request}
    """
)
def code_request(request: str) -> str: ...


# Parsy Test Functions ----------------------------------------------------------------------------------------------
goal = (
    "I want to create a highly flexible parsing library using the Python Parsy library."
)

genre = "fantasy"

genre_list = [
    "sci-fi",
    "action",
    "adventure",
    "fantasy",
    "mystery",
    "romance",
    "non-fiction",
    "biography",
    "historical",
    "comedy",
    "historical fiction",
    "science",
    "self-help",
    "dystopian",
    "utopian",
]
# goal = "I want to create a framework of pydantic classes that represent the software product development process, in order to represent the process in code."
# goal = "I want to create a file management library that provides directory and file reading/writing/creation/renaming capabilities all from a Pydantic baseclass. There should be an inherited class to represent directories, and another inherited class to represent files. Keep as much functionality within the baseclass as possible."


# goal = "I want to create a TUI for viewing files using the Textual Python library. It should consist of a Text Area and a sidebar with another text area and a submission button for inputting text. If a filename is input to the text area and the button is clicked, the text area should read the document from the file and display them in the text area. There should be a second 'save' button in the sidebar that saves any edits made in the text area back to file."


def gen_pydantic_ideas(goal):
    reqs = get_requirements(goal)
    milestones = []
    requirement_list = []
    print(f"\nSystem Requirements:")
    for req in reqs.requirements:
        print(f"    {req.requirement_name}:\n        {req.requirement_description}")
        requirement_list.append(req)
    condensed_functionality = condense_functionality(goal, reqs)  # requirement_list)
    print(f"\n\nCondensed Functionality:\n")
    # print(f"\n\n{condensed_functionality}\n\n")
    for func in condensed_functionality.functionality_list:
        print(
            f"    Functionality Group: {func.functionality_name}\n                   Desc: {func.functionality_description}\n"
        )

    count = 0
    for req in condensed_functionality.functionality_list:
        count += 1
        print(f"\nGenerating Functional Requirement Milestones {count}")
        milestone_list = create_milestones(goal, req)
        milestones.append({req.functionality_name: milestone_list})
    print(f"\n\nMilestones:")
    for milestone in milestones:
        for milestone_name, milestone_list in milestone.items():
            print_breakline()
            print(f"\n{milestone_name} milestones:\n")
            print_milestone_requirements(milestone_list)
    class_ideas = get_class_ideas(goal, condensed_functionality)
    print(f"\n\nClass Ideas:")
    for idea in class_ideas.idea_list:
        print(f"    {idea.description}")
    pydantic_models = []
    # print(f"\n\n")
    for idea in class_ideas.idea_list:
        pydantic_class = pydantic_request(
            goal, condensed_functionality, idea.description
        )
        pydantic_models.append(pydantic_class)
    # for model in pydantic_models:
    #    print(f"\n{model}\n")
    for model in pydantic_models:
        print_pydantic_model(model)


def remove_id_fields(data):
    if isinstance(data, dict):
        return {k: remove_id_fields(v) for k, v in data.items() if k != "id"}
    elif isinstance(data, list):
        return [remove_id_fields(item) for item in data]
    return data


def gen_book(genre: str, current_list: list[str] = []):
    # gen_pydantic_ideas(goal)
    # print(f"\nRequesting a {genre} book recommendation...\n")
    example_id = "bd3fcdbfae104704ad04e32b931888eb"
    example = get_single_instance_from_db("Book", example_id)
    example = example.model_dump()
    example = remove_id_fields(example)
    # rprint(example)
    try:
        # book = test_pydantic_request(num=3, genre=genre)
        book = test_pydantic_request(
            genre=genre, current_list=current_list, example=example
        )

        # print(f"\nBook Reccommendation: \n")  # "\n{book}\n\n")
        # rprint(book)
        return book
        # print(f"\n\n")
    except ValidationError as e:
        print(f"\n\nError: {e}\n\n")
        response = cast(openai.OpenAICallResponse, e._response.content)  # pyright: ignore[reportAttributeAccessIssue]
        rprint(response)
        return f"Error with genre: {genre}"
    # print(f"\n\nBook Reccommendation: \n\n{book}\n\n")
    # for res in book:
    #    print(f"\n{res}\n")
    # print(f"\n\n")
    # return book


def gen_book_list(num: int, genre: str):
    # gen_pydantic_ideas(goal)
    # print(f"\nRequesting a {genre} book recommendation...\n")
    id = "640ee6f910774ecbbb64e848c4ce342a"
    example = get_single_instance_from_db("BookList", id)
    example = example.model_dump()
    example = remove_id_fields(example)
    # rprint(example)
    try:
        # book = test_pydantic_request(num=3, genre=genre)
        book = test_book_list(num=num, genre=genre, example=example)

        # print(f"\nBook Reccommendation: \n")  # "\n{book}\n\n")
        # rprint(book)
        return book
        # print(f"\n\n")
    except ValidationError as e:
        print(f"\n\nError: {e}\n\n")
        response = cast(openai.OpenAICallResponse, e._response)  # pyright: ignore[reportAttributeAccessIssue]
        rprint(response.model_dump())
        return f"Error with genre: {genre}"
    # print(f"\n\nBook Reccommendation: \n\n{book}\n\n")
    # for res in book:
    #    print(f"\n{res}\n")
    print(f"\n\n")
    # return book


def check_uniques(book_list: list[Book]):
    unique_books = []
    all_titles = []
    unique_titles = []
    print(f"\n\n")
    for book in book_list:
        all_titles.append(book.title)
        print(f"{book.title}")
    print(f"\n\n")
    for book in book_list:
        if book.title not in unique_titles:
            unique_titles.append(book.title)
            unique_books.append(book)
            print(f" - '{book.title}'")
    return unique_books


## Code Generation --------------------------------------------------------------------------------------------------
# phi3.5                    | Generated code in 18.11 sec (0.3 min)
# phi3.5                    | Generated   56 lines of code in 14.87 sec (0.2 min)
# qwen2.5-coder:7b-instruct | Generated   69 lines of code in 38.89 sec (0.6 min)
# Mistral Small             | Cloudflare tunnel timeout
# codestral                 | Generated    5 lines of code in 164.31 sec (2.7 min) - Absolute garbage, just said what it'll do.


def gen_project_structure():
    start_time = time()
    code_test = "Create a python script that flexibly parses JSON responses to a given Pydantic model. The script is primarily intended to recover from small mistakes in structure and formatting. The script should respond with the Pydantic model object and True if the JSON is successfully parsed, and an error message and False if the JSON is not successfully parsed. Keep all description and explanation contained within code comments and docstrings. *DO NOT* include any preable or overview. *ONLY* return valid python code."

    result = code_request(code_test)
    print(type(result))
    code_lines = len(result.content.splitlines())
    end_time = time()
    print(f"\n\nCode Result: \n\n{result}\n\n")
    elapsed_time = end_time - start_time
    rounded_sec = round(elapsed_time, 2)
    rounded_min = round(elapsed_time / 60, 1)
    print(
        f"\n# {local_model_name:<25} | Generated {code_lines:>4} lines of code in {rounded_sec:>5} sec ({rounded_min:>3} min) "
    )


## Pydantic Book generation ------------------------------------------------------------------------------------------------
# Llama 3.2 = 85 in 107.8 sec
# Mistral Nemo = 85 in 967.1 sec - 1 error
# Mistral Nemo = 150 5 book lists in 8764.6 sec - 2 errors
# Qwen2.5 coder = 75 in 297.5 sec - 0 errors
# llama3.2:3b-instruct-q4_K_M = 75 books in 102.5 sec - 37 errors
# llama3.2:3b-instruct-q6_K = 75 books in 123.0 sec - 22 errors
# Gemma2:2b - Failed to complete
# llama3.2:3b-instruct-q8_0 = 75 books in 119.4 sec - 13 errors
# llama3.2:3b-instruct-q8_0 = 150 books in 255.2 sec - 26 errors
# gemma2:2b = 150 books in 278.8 sec - 9 errors
# mistral-nemo = 75 books in 901.9 sec - 2 errors
# gemma2:2b = 45 books in 91.0 sec - 3 errors
# gemma2:2b-instruct-q8_0 = 45 books in 127.4 sec - 18 errors
# gemma2:2b = 45 books in 101.2 sec - 3 errors (with example from db as template)
# llama3.2:3b-instruct-q8_0 = 45 books in 98.4 sec - 18 errors (with example from db as template)
# llama3.2:3b-instruct-q8_0 = 45 books in 82.9 sec - 5 errors (with example.model_dump() from db as template)
# gemma2:2b = 45 books in 107.2 sec - 3 errors (with example.model_dump() from db as template)
# gemma2:2b = 150 books in 304.1 sec - 3 errors
# granite3-dense:2b = 75 books in 164.0 sec - 0 errors (with example.model_dump() from db as template)
# granite3-dense:2b = 75 books in 129.8 sec - 0 errors (no example, just a response model. Dayum.)
# granite3-dense:2b = 30 books in 57.0 sec | 0.5 books/sec | 3.33% error | 1 total errors
# granite3-dense:2b = 30 books in 218.8 sec | 0.1 books/sec | 13.33% error | 4 total errors  (Book list - just spat input back)
# granite3-dense:2b = 300 books in 525.6 sec | 0.6 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# granite3-dense:2b = 150 books in 324.0 sec | 18.0% unique | 0.5 books/sec | 0.67% error | 1 total errors  ==> (Book gen)
# granite3-dense:8b = 75 books in 476.6 sec | 32.0% unique | 0.2 books/sec | 1.33% error | 1 total errors  ==> (Book gen)
# qwen2.5-coder:7b-instruct = 75 books in 323.9 sec | 49.33% unique | 0.2 books/sec | 1.33% error | 1 total errors  ==> (Book gen/list)
# gemma2:2b = 75 books in 246.7 sec | 17.33% unique | 0.3 books/sec | 73.33% error | 55 total errors  ==> (Book gen - No template)
# llama3.2:3b-instruct-q6_K = 75 books in 138.9 sec | 37.33% unique | 0.5 books/sec | 2.67% error | 2 total errors  ==> (Book gen/list)
# llama3.2:3b-instruct-q8_0 = 75 books in 166.4 sec | 32.0% unique | 0.5 books/sec | 9.33% error | 7 total errors  ==> (Book gen/list)
# gemma2:2b-instruct-q8_0 = 75 books in 204.6 sec | 26.67% unique | 0.4 books/sec | 17.33% error | 13 total errors  ==> (Book gen/list)
# mistral-nemo = 45 books in 633.8 sec | 86.67% unique | 0.1 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# mistral-small = 45 books in 1511.9 sec | 97.78% unique | 0.0 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# phi3.5 = 45 books in 177.1 sec (3.0 min) | 73.33% unique | 0.3 books/sec | 2.22% error | 1 total errors  ==> (Book gen/list)
# phi3.5 = 75 books in 185.7 sec (3.1 min) | 58.67% unique | 0.4 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# phi3.5 = 150 books in 378.5 sec (6.3 min) | 26.0% unique | 0.4 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# mistral-nemo = 75 books in 1103.8 sec (18.4 min) | 81.33% unique | 0.1 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# phi3.5                    =    75 books in 191.8 sec (3.2 min) | 56.0% unique | 0.4 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# gemma2:2b                 =    75 books in 189.6 sec (3.2 min) | 38.67% unique | 0.4 books/sec | 18.67% error | 14 total errors  ==> (Book gen/list)
# llama3.2:3b-instruct-q8_0 =    75 books in 136.7 sec (2.3 min) | 36.0% unique | 0.5 books/sec | 8.0% error | 6 total errors  ==> (Book gen/list)
# llama3.2:3b-instruct-q8_0 =    75 books in 135.5 sec (2.3 min) | 37.3% unique | 0.6 books/sec | 1.33% error | 1 total errors  ==> (Book gen/list)
# granite3-dense:2b         =    75 books in 165.0 sec (2.8 min) | 28.0% unique | 0.5 books/sec | 0.0% error | 0 total errors  ==> (Book gen/list)
# mistral-small             =   375 books in 30246.2 sec (504.1 min) | 42.4% unique | 0.0 books/sec | 2.93% error | 11 total errors  ==> (Book gen/list)
# qwen2.5-coder:7b-instruct =   375 books in 1658.64 sec (27.6 min) | 85 unique (22.7%) | 0.2 books/sec | 0.8% error | 3 total errors  ==> (Book gen - Lots of bullshit)
# mistral-nemo              =   150 books in 2402.53 sec (40.0 min) | 111 unique (74.0%) | 0.1 books/sec | 0.0% error | 0 total errors  ==> (Book gen - Great responses)


def gen_project_structure2():
    start_time = time()
    overall_book_list = []
    for i in range(10):
        overall_book_list.extend(genre_list)
    results = []
    count = 0
    # print(f"\n\n")
    result_title = []
    for genre in overall_book_list:
        count += 1
        print(f"Request {count}: Requesting a {genre} book recommendation...")
        return_book = gen_book(genre, result_title)
        results.append(return_book)
        result_title.append(return_book.title)
        # results.append(gen_book_list(5, genre))
    print(f"\n\n")
    count = 0
    error_results = []
    successful_results = []
    for res in results:
        count += 1
        print(f"\nResult {count} - Type: {type(res)}\n")
        rprint(res)
        if isinstance(res, str):
            error_results.append({"Error": res})
        else:
            successful_results.append(res)  # Single
            ## List:
            # for book in res.books:
            #    successful_results.append(book)
    parsed_results = []
    """
    for res in results:
        try:
            pydantic_model = Book.model_validate_json(res)
            # pydantic_model = BookList.parse_obj(res)
            parsed_results.append(pydantic_model)
        except ValidationError as e:
            error_results.append({"Error": e, "response": res})
    print(f"\nParsed Results:\n")
    for res in parsed_results:
        rprint(res)
    print(f"\nError Results:\n")
    for res in error_results:
        rprint(res)
    """
    unique_results = check_uniques(successful_results)
    end_time = time()
    elapsed_time = end_time - start_time
    rounded_sec = round(elapsed_time, 2)
    rounded_min = round(elapsed_time / 60, 1)
    # print(f"\n{len(parsed_results)} parsed results.\n")
    # print(f"\n{len(error_results)} errors.\n")
    # print(
    #    f"\n\n\nFinished gnerating {len(overall_book_list)} books in {rounded_sec} seconds.\n"
    # )
    book_list_len = len(overall_book_list)
    bps = round(book_list_len / rounded_sec, 1)
    unique_percentage = round((len(unique_results) / book_list_len) * 100, 1)
    errors = len(error_results)
    error_percentage = (errors / book_list_len) * 100
    rounded_error_percentage = round(error_percentage, 2)
    # print(f"\n\n\n")
    # for u_result in unique_results:
    #    print(f" - '{u_result.title}'")
    # print(f"\n")
    print(
        f"\n\n\n# {local_model_name:<25} = {len(overall_book_list):>5} books in {rounded_sec:>5} sec ({rounded_min:>3} min) | {len(unique_results)} unique ({unique_percentage}%) | {bps} books/sec | {rounded_error_percentage}% error | {len(error_results)} total errors  ==> (Book gen/list)\n"
    )
    print(f"\n{db_url}\n")


def test_pydantic():
    print(f"Generating pydantic class...")
    pydantic_result = pydantic_request()
    print(f"\n\n{pydantic_result}\n\n")


def sample_parsy_request():
    print(f"\nTesting Parsy Request: {input_req_1}\n")
    test_result = parsy_request(input_req_1, expected_output_1)
    # print(f"\nResult: {test_result}\n")
    print(f"\nDescription: {test_result.description}\n")
    parsed_output = parse_output_code(test_result.parse_function_code)
    # print(f"\nOutput: \n{parsed_output}\n")
    return test_result.parse_function_code


def test_parsy_request():
    output_code = sample_parsy_request()
    import_str = f"import parsy as p\nfrom parsy import generate\n\n"
    # print(f"\n\n\nOutput Code: \n\n{import_str}{output_code}\n\n")
    # parser_obj = eval(import_str + output_code)
    # print(f"\nParser Object: {parser_obj}\n")
    test_parser = GeneratedParser(
        input="1992-12-31",
        llm_output_code=output_code,
    )
    test_parser.format_parser()
    # print(f"\nTest Result: {test_parser.result_msg}\n")
    # print(f"\nTest Output: {test_parser.parsed_output}\n")

    # print(f"\nOutput Code: {output_code}\n")


# Functions ---------------------------------------------------------------------------------------------------------


def print_class_attributes(obj, indent=""):
    if not hasattr(obj, "__dict__"):
        print(f"{indent}{obj}")
        return

    for attr_name, attr_value in vars(obj).items():
        print(f"{indent}{attr_name}:")
        if isinstance(attr_value, (int, float, bool, str, type(None))):
            print(f"{indent}  {attr_value}")
        else:
            print_class_attributes(attr_value, indent + "  ")


def break_long_lines(text, max_length=80):
    lines = text.split("\n")
    result = []

    for line in lines:
        if len(line) <= max_length:
            result.append(line)
        else:
            current_line = ""
            words = line.split()

            for word in words:
                if len(current_line) + len(word) + 1 <= max_length:
                    current_line += " " + word if current_line else word
                else:
                    result.append(current_line)
                    current_line = word

            if current_line:
                result.append(current_line)

    return "\n".join(result)


def print_class_attributes(obj, tree=None, indent=""):
    if tree is None:
        tree = Tree(f"[bold magenta]{type(obj).__name__}[/bold magenta]")

    if not hasattr(obj, "__dict__"):
        tree.add(f"[yellow]{repr(obj)}[/yellow]")
        return tree

    for attr_name, attr_value in vars(obj).items():
        if isinstance(attr_value, (int, float, bool, type(None))):
            tree.add(
                f"[underline][cyan]{attr_name}[/cyan]: [yellow]{repr(attr_value)}[/yellow][/underline]"
            )
        elif isinstance(attr_value, str):
            string_tree = tree.add(
                f"[underline][cyan]{attr_name}[/cyan] ([green]str[/green][/underline]):"
            )
            split_string = break_long_lines(attr_value)
            for line in split_string.split("\n"):
                syntax = Syntax(
                    f"{line}", "python", theme="monokai", line_numbers=False
                )
                string_tree.add(syntax)
        elif isinstance(attr_value, (list, tuple, set)):
            collection_tree = tree.add(
                f"[underline][cyan]{attr_name}[/cyan] ([green]{type(attr_value).__name__}[/green][/underline]):"
            )
            for item in attr_value:
                print_class_attributes(item, collection_tree, indent + "  ")
        elif isinstance(attr_value, dict):
            dict_tree = tree.add(
                f"[underline][cyan]{attr_name}[/cyan] ([green]dict[/green][/underline]):"
            )
            for key, value in attr_value.items():
                key_tree = dict_tree.add(f"[cyan]{repr(key)}[/cyan]:")
                print_class_attributes(value, key_tree, indent + "  ")
        elif hasattr(attr_value, "__dict__"):
            subtree = tree.add(
                f"[underline][cyan]{attr_name}[/cyan] ([green]{type(attr_value).__name__}[/green][/underline]):"
            )
            print_class_attributes(attr_value, subtree, indent + "  ")
        else:
            tree.add(
                f"[underline][cyan]{attr_name}[/cyan]: [yellow]{repr(attr_value)}[/yellow][/underline]"
            )

    return tree


def display_object(obj):
    console = Console()
    tree = print_class_attributes(obj)
    console.print(tree)


def process_class_attributes(obj):
    if not hasattr(obj, "__dict__"):
        return obj

    for attr_name, attr_value in vars(obj).items():
        if isinstance(attr_value, str):
            setattr(obj, attr_name, attr_value.replace("\n", "\\n"))
        elif isinstance(attr_value, (int, float, bool, type(None))):
            continue
        else:
            setattr(obj, attr_name, process_class_attributes(attr_value))

    return obj


# Recursively replace all instances of a string in all attributes of a class
def recursive_replace(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)
    elif isinstance(obj, dict):
        return {k: recursive_replace(v, old, new) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_replace(elem, old, new) for elem in obj]
    else:
        for attr in obj.__dict__:
            setattr(obj, attr, recursive_replace(getattr(obj, attr), old, new))
            return obj


# Print the llm output, add an additional slash to newlines:
def print_output(output):
    printable_output = process_class_attributes(
        output
    )  # recursive_replace(output, "\n", "\\n")
    rprint(printable_output)


# Function to store the output class name and id of an llm call for later call from db
def store_output(output_list: list, output: MiraResponse):
    logger.info(f"Software Architect | Storing Output: \n\n{output}\n")
    step = ArchitectAskStep(
        response_type=output.__class__.__name__,
        response_id=output.id,
    )
    output_list.append(step)
    return output_list


def write_code_file(
    filename_seed: str,
    # start_time: str,
    directory: str,
    initial_request: str,
    content: str,
):
    """
    Write content to a file in the specified directory.

    Args:
        filename (str): The name of the file to be created.
        directory (str): The directory where the file should be saved.
        content (str): The content to be written to the file.

    Returns:
        str: The full path of the created file.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    filename = filename_seed + ".py"  # + "_" + start_time + ".py"
    # Create the full file path
    file_path = os.path.join(directory, filename)
    output_content = (
        "# A python file generated via llm to solve the following request: \n"
        + '"""\n'
        + initial_request
        + '\n"""\n\n'
        + content
    )
    # Write the content to the file
    with open(file_path, "w") as file:
        file.write(output_content)

    return file_path


def strip_directory(filepath: str):
    file_name = filepath.split("/")[-1]
    return file_name


def parse_filename(filepath: str):
    filename = strip_directory(filepath)
    file = filename.split(".")[0]
    return file


def test_code_request(code_output_dir: str):
    # code_output_dir = "code_output/"
    start_time = datetime.now()
    start_time_str = start_time.strftime("%y%m%d_%H%M%S")
    code_output_dir = code_output_dir + start_time_str + "/"
    code_monkey = SoftwareArchitect(request=code_request, output_dir=code_output_dir)
    result = code_monkey.ask()
    return result


# Classes -----------------------------------------------------------------------------------------------------------


class ArchitectAskStep(BaseModel):
    response_type: str
    response_id: str


class ArchitectAskResult(BaseModel):
    query: str
    step_response_list: List[ArchitectAskStep]


class ArchitectAskList(BaseModel):
    queries: List[ArchitectAskResult]


# Architect Class
class ParsingArchitect(BaseModel):
    request: Optional[str] = None
    output_dir: Optional[str] = None
    intermediate_steps: Optional[list[IntermediateResponse]] = []
    output_code: Optional[str] = None
    test_cases: Optional[TestLibrary] = None
    test_suite: Optional[str] = None

    # @log_output(logger, state_log)
    def ask(self, code_request: str = "", code_output_dir: str = ""):
        start_time = time()
        if code_request == "":
            if not self.request:
                raise ValueError(
                    "There is no request in the function or the class. Please try again."
                )
        else:
            if code_request != self.request:
                self.request = code_request
        if code_output_dir == "":
            if not self.output_dir:
                raise ValueError(
                    "There is no code output directory in the function or the class. Please try again."
                )
        else:
            if code_output_dir != self.output_dir:
                self.output_dir = code_output_dir

        step_ids = []
        logger.info(f"Software Architect | Asking query: {self.request}")
        initial_code = self.self_ask(self.request)
        logger.info(f"Software Architect | Self Ask Result: \n\n{initial_code}\n")

        step_ids = store_output(step_ids, initial_code)
        logger.warning(
            f"Software Architect | Self Ask Result: {type(initial_code)}\n\n"
        )
        initial_file = write_code_file(
            "initial", self.output_dir, self.request, initial_code.response.code.code
        )
        finished_response = self.finish_code_example(
            self.request, initial_code.response.code.code
        )
        step_ids = store_output(step_ids, finished_response)
        logger.warning(
            f"Software Architect | Finished Code Example Response: {type(finished_response)}\n"
        )
        final_file = write_code_file(
            "final",
            self.output_dir,
            self.request,
            finished_response.code.code,
        )
        self.output_code = finished_response.code.code
        logger.info(
            f"Software Architect | Generating Tests from code: {self.output_code}\n"
        )
        test_response = self.generate_tests(self.request, finished_response.code.code)
        logger.info(f"Software Architect | Tests Generated: \n\n{test_response}\n")
        step_ids = store_output(step_ids, test_response)
        test_case_list = []
        test_data_list = []
        self_test_cases = []
        for test in test_response.test_cases:
            for data in test.test_data:
                test_data_list.append(data)
            test_case_list.append(test.test_case.code)
            self_test_cases.append(test.test_case)

        # self.test_data = test_data_list
        self.test_cases = test_response

        final_test_library = self.generate_test_library(
            self.request, finished_response.code.code, test_case_list, test_data_list
        )
        step_ids = store_output(step_ids, final_test_library)
        logger.warning(f"Software Architect | Test Library: {type(final_test_library)}")
        try:
            replaced_lib_import_text = final_test_library.code.code.replace(
                "#IMPORT_LIB", ".final"
            )
        except:
            replaced_lib_import_text = final_test_library.code.code

        final_file = write_code_file(
            "tests",
            self.output_dir,
            self.request,
            replaced_lib_import_text,  # final_test_library.code.code
        )

        self.test_suite = replaced_lib_import_text
        final_file_str = final_file.replace("/", ".")
        end_time = time()
        elapsed_time = end_time - start_time
        rounded_sec = round(elapsed_time, 1)
        logger.info(
            f"SoftwareArchitect | Ask | Finished Ask loop in {rounded_sec} seconds."
        )
        query_response = ArchitectAskResult(
            query=code_request, step_response_list=step_ids
        )
        return query_response  # step_ids

    def lookup_result(self, response_step: ArchitectAskStep):
        result = get_single_instance_from_db(
            response_step.response_type, response_step.response_id
        )
        return result

    # Regular Self Ask
    # @log_output(logger, state_log)
    @miracall_self_ask_dec
    def self_ask(
        self,
        query: str,
        examples: list[FewShot] = few_shot_code_examples,
    ) -> openai.OpenAIDynamicConfig:
        return {
            "computed_fields": {
                # "examples": [
                #    [example["question"], example["answer"]] for example in examples
                # ]
                "examples": [[example.question, example.answer] for example in examples]
            }
        }

    # @log_output(logger, state_log)
    @miracall_gen_test_library_dec
    def generate_test_library(
        self, goal: str, code: str, test_cases: list[str], test_data: list[TestData]
    ): ...

    # @log_output(logger, state_log)
    @miracall_gen_tests_dec
    def generate_tests(self, goal: str, code: str): ...

    # @log_output(logger, state_log)
    @miracall_finish_code_dec
    def finish_code_example(self, goal: str, code: str): ...


test_instances = [
    ["07e6b8df7f774f32a24bff2fc2a08ea0", "CodeResponse"],
    ["8079c46fcac54eb38813d26b7f65ef3c", "CodeResponse"],
    ["226192648c4c494fafbf5de05d2db552", "TestLibrary"],
    # [SelfAskCodeResponse, 272636118fe24469818940b488819b13],
    # [CodeResponse, 4b5f8e0780c64072bffb1b8fb1a388c6],
    # [TestLibrary, 6cc168768c9648dd858b84bb2332a383],
    # [CodeResponse, 775fa2e3334a4ff99e85cf5e1318ff26],
    ["142cf3e1456241f3a13d466ada8198cc", "SelfAskCodeResponse"],
    ["216f035ef88248f19ad52c448d907933", "CodeResponse"],
    ["b3d888abd5c4411c9f8208c601138bc8", "TestLibrary"],
    ["2c87d94c48364f40bf8df8abf11fff3b", "CodeResponse"],
    ["b04c9936db96468eb30194602a967070", "SelfAskCodeResponse"],
    ["4d3efff88c774bfd825c3d6bbeed718c", "CodeResponse"],
    ["89e7fddb3de74f19b17f961dd19754d8", "TestLibrary"],
    ["01e2e16b47ae4ab8bd749055ceab112d", "CodeResponse"],
]


def existing_result_test():
    for test in test_instances:
        result = get_single_instance_from_db(test[1], test[0])
        print(f"\nResult Type: {type(result)}\n")
        display_object(result)
        print("\n\n")
    print(f"\n{db_url}\n\n")


# if __name__ == "__main__":
def new_test():
    print(f"Testing code request:\n\nCode Output Directory: {code_output_dir}\n")
    result = test_code_request(code_output_dir)
    print(f"\nResult Type: {type(result)}\n")
    rprint(result)
    result_list = []
    print("\n\n")
    # processed_result = process_class_attributes(result)
    display_object(result)

    print(f"\nResult:  {result}\n\n")
    for step in result.step_response_list:
        #    # print(f"\n{step}\n")
        instance_step = [step.response_id, step.response_type]
        result = get_single_instance_from_db(step.response_type, step.response_id)
        print(f"\nResult Type: {type(result)}\n")
        #    rprint(result)
        #    print("\n\n")
        #    # processed_result = process_class_attributes(result)
        display_object(result)
        #    # print_output(result)
        #    # print_class_attributes(result)
        result_list.append(instance_step)
        print("\n\n")

    for step in result_list:
        step_str = f"    ['{step[0]}', '{step[1]}'],"
        print(step_str)

    print(f"\n{db_url}\n\n")


def iterate_prompts(container: PromptContainer):
    print(f"\nContainer: {container}\n")
    for prompt in container.prompts:
        print(f"Prompt {type(prompt.name)} -> {prompt.name}")
        print(
            f"Prompt {type(prompt.prompt.prompt)} -> \n        {prompt.prompt.prompt}"
        )
        print(
            f"Prompt {type(prompt.prompt.prompt_type)} -> {prompt.prompt.prompt_type}\n"
        )
        # print(f"\nPrompt: {prompt.prompt}\n")
        # print(f"\nPrompt: {prompt.system_prompt}\n")
        # print(f"\nPrompt: {prompt.user_prompt


def test():
    # new_test()
    existing_result_test()
    dump = architect_prompts.model_dump_json()
    print(f"\n\nTest Prompt: {type(architect_prompts)}\n\n{dump}\n\n")
    print(f"Test Prompt Dump:\n")
    for key, value in json.loads(dump).items():
        print(f"\n - {key}: \n{value}")

    print(
        f"\n\nArchitect Prompts: Type: {type(architect_prompts)}\n\n{architect_prompts}\n\n"
    )
    # for attr in test_markdown_section_prompt.model_dump_json():
    #    print(f"\n{attr}: {getattr(test_markdown_section_prompt, attr)}\n")
    # for prop in test_markdown_section_prompt
    #    print(f"\n{prop}: {getattr(test_markdown_section_prompt, prop)}\n")
    iterate_prompts(user_prompts)
    print(f"\n\n")
    iterate_prompts(system_prompts)
    prompt = system_prompts.test_library_system
    print(
        f"\n\nACCESSING PROMPT BY DOT NOTATION: {type(prompt)}\n\n{prompt}"  # ".test_library_system}\n"
    )
    # print(f"\n{test_markdown_section_prompt.system_prompt}\n")
    # print(f"\n{test_markdown_section_prompt.user_prompt}\n")
    # print(f"\n{test_markdown_section_prompt.prompt}\n")
    print(f"\n\n{db_url}\n\n")


# Misc
