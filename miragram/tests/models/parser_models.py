from typing import TypeVar, Type, Any, get_type_hints, Dict, Generic, Optional
from pydantic import BaseModel, create_model
from functools import reduce
import parsy as P
from parsy import generate, regex, string, whitespace, Parser as ParsyParser

from abc import ABC, abstractmethod


class Parser(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def generated_parser(cls) -> ParsyParser | None:
        """Override this method to provide a generator-based parser implementation"""
        return None

    @classmethod
    def parser(cls) -> ParsyParser:
        """Returns the parser implementation for this class"""
        generated = cls.generated_parser()
        if generated:
            return generated
        raise NotImplementedError(
            "Must implement either parser() or generated_parser()"
        )

    @classmethod
    def parse(cls, text: str):
        parser = cls.parser()
        result = parser.parse(text)
        return cls(**result)

    # @abstractmethod
    # def generate(self) -> str:
    #    """Generate the string representation of this parsed element"""
    #    pass


class GeneratedParser(BaseModel):
    input: str
    expected_output: Optional[Any] = None
    result_msg: Optional[str] = None
    llm_output_code: str
    created_parser: Optional[P.Parser] = None
    parsed_output: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def format_parser(self):
        print(f"\nParser for {self.input}:\n\n{self.llm_output_code}")
        parsed = self.rebuild_parser()
        print(f"\n\nParsed output: \n{parsed}\n\n")

        evaluated = exec(parsed).parse(self.input)
        print(f"\n\nEvaluated: {evaluated}\n\n")

    def rebuild_parser(self):
        parser_components, rest = parse_decorator_and_function.parse_partial(
            self.llm_output_code
        )
        print(f"\n\nParsed output: {parser_components}\n\n")
        parser_str = f"@{parser_components['decorator_name']}\ndef {parser_components['function_name']}{rest}"
        print(f"\n\nRebuilt parser: \n\n{parser_str}\n\n")
        return parser_str

    def test(self):
        try:
            parser = self.created_parser
            result = parser.parse(self.input)
            self.parsed_output = result
            if self.expected_output:
                assert result == self.expected_output
                self.result_msg = "Success"
        except P.ParseError as e:
            self.result_msg = f"Error parsing {self.input}: {e}"
            raise AssertionError(f"Error parsing {self.input}: {e}")


@generate
def parse_decorator_and_function():
    # Parse the @ symbol and decorator name
    yield string("@")
    decorator_name = yield regex(r"[a-zA-Z_][a-zA-Z0-9_]*")

    """
    # Handle optional decorator arguments
    yield whitespace.many()
    open_paren = yield string("(").optional()

    if open_paren:
        args = yield (
            regex(r"[^)]+")  # Match everything until closing parenthesis
            | whitespace.many().result("")  # Or empty for no args
        )
        yield string(")")
        decorator_full = f"@{decorator_name}({args})"
    else:
        decorator_full = f"@{decorator_name}"
    """
    # Handle whitespace and newlines between decorator and function
    yield whitespace.many()

    # Parse the function definition
    yield string("def ")
    func_name = yield regex(r"[a-zA-Z_][a-zA-Z0-9_]*")
    """
    # Parse function parameters
    yield string("(")
    params = yield regex(r"[^)]*")  # Match everything until closing parenthesis
    yield string(")")

    # Handle function body
    yield whitespace.many()
    yield string(":")
    yield whitespace.many()

    # Parse function body (indented block)
    function_body_str = yield (regex(r"[ ]{4}[^\n]+\n") | regex(r"\t[^\n]+\n"))

    # indented_line = (
    #    regex(r"    [^\n]*\n").pipe(  # Match 4-space indented lines
    #        regex(r"\t[^\n]*\n")
    #    )  # Or tab-indented lines
    # )
    # function_body = yield indented_line.at_least(1)
    # function_body_str = "".join(function_body)
    function_full = f"def {func_name}({params}):\n{function_body_str}"
    """
    return {
        # "decorator": decorator_full,
        # "function": function_full,
        "decorator_name": decorator_name,
        "function_name": func_name,
    }
