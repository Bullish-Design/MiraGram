# Imports
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from sqlalchemy.engine import result
from sqlmodel import SQLModel, Relationship, Field as SQLField
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import os
from time import time
from datetime import datetime
from rich.console import Console
from rich.tree import Tree
from rich.syntax import Syntax
from rich import print as rprint

from mirascope.core import openai


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
from miragram.tests.functions.code import (
    FewShot,
    TestLibrary,
    TestData,
    few_shot_code_examples,
    self_ask_dec,
    finish_code_dec,
    gen_tests_dec,
    gen_test_library_dec,
    miracall_self_ask_dec,
    miracall_finish_code_dec,
    miracall_gen_tests_dec,
    miracall_gen_test_library_dec,
)

# Configure logging
from miragram.src.base.config import code_output_dir, db_url

from miragram.log.logger import get_logger

logger = get_logger("SoftwareArchitect_Logging")


# Constants ---------------------------------------------------------------------------------------------------------
code_request = """
    Please create a python script that takes a folder directory as argument. Inside that directory is 3 sub directories: "code", "data", and "tests". The python script should run all pytest functions inside the "tests" directory by running the code inside the "code" directory against all data in the "data" directory. 
    """
code_output_dir = code_output_dir

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
class SoftwareArchitect(BaseModel):
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


def test():
    new_test()
    # existing_result_test()


# Misc
