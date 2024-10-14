# Imports
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from sqlmodel import SQLModel, Relationship, Field as SQLField
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import os
from time import time
from datetime import datetime

# Local Imports
from miragram.src.base.base import (
    IntermediateResponse,
    ArchitectAskResult,
    ArchitectAskStep,
    MiraResponse,
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
    get_single_instance_from_db,
)

# Configure logging
from miragram.log.logger import get_logger

logger = get_logger("SoftwareArchitect_Logging")


# Constants ---------------------------------------------------------------------------------------------------------
code_request = """
    Please create a python script that takes a folder directory as argument. Inside that directory is 3 sub directories: "code", "data", and "tests". The python script should run all pytest functions inside the "tests" directory by running the code inside the "code" directory against all data in the "data" directory. 
    """


# Functions ---------------------------------------------------------------------------------------------------------


# Function to store the output class name and id of an llm call for later call from db
def store_output(output_list: list, output: MiraResponse):
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


def test_code_request():
    code_output_dir = "code_output/"
    start_time = datetime.now()
    start_time_str = start_time.strftime("%y%m%d_%H%M%S")
    code_output_dir = code_output_dir + start_time_str + "/"
    code_monkey = SoftwareArchitect(request=code_request, output_dir=code_output_dir)
    result = code_monkey.ask()
    return result


# Classes -----------------------------------------------------------------------------------------------------------


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
            f"Software Architect | Finished Response: {type(finished_response)}\n\n"
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
        logger.warning(
            f"Software Architect | Test Library: {type(final_test_library)}\n\n"
        )
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
    @self_ask_dec
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
    @gen_test_library_dec
    def generate_test_library(
        self, goal: str, code: str, test_cases: list[str], test_data: list[TestData]
    ): ...

    # @log_output(logger, state_log)
    @gen_tests_dec
    def generate_tests(self, goal: str, code: str): ...

    # @log_output(logger, state_log)
    @finish_code_dec
    def finish_code_example(self, goal: str, code: str): ...


if __name__ == "__main__":
    test_code_request()
