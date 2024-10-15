# Imports -------------------------------------------------------------------------------------------------------------
from pydantic import (
    BaseModel,
    PrivateAttr,
    model_validator,
    field_validator,
    computed_field,
    Field,
    validator,
)
from typing import List, Optional, Dict, Type  # , TypeOf
import re
from enum import StrEnum, auto
# Local imports -------------------------------------------------------------------------------------------------------


# Logging -------------------------------------------------------------------------------------------------------------
from miragram.log.logger import get_logger

logger = get_logger("PromptBase")

# Constants -----------------------------------------------------------------------------------------------------------
system_header = """
    SYSTEM:
    """

user_header = """
    USER:
    """


# Functions -----------------------------------------------------------------------------------------------------------
def extract_bracketed_text(text):
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, text)
    return matches


# Classes -------------------------------------------------------------------------------------------------------------
class PromptType(StrEnum):
    USER = auto()
    SYSTEM = auto()
    MODIFIER = auto()

    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class PromptInstance(BaseModel):
    prompt: str = ""
    prompt_type: PromptType = PromptType.USER


class IndividualPrompt(BaseModel):
    prompt: PromptInstance = PromptInstance()
    name: str = ""

    # def __init__subclass__(cls, **kwargs):
    #    super().__init_subclass__(**kwargs)
    #    logger.info(f"IndividualPrompt subclass created: {cls.__name__}")
    #    logger.info(f"Name: {self.__name__}")

    def model_post_init(self, __context):
        logger.info(f"IndividualPrompt instance created: {self}")
        logger.info(f"Dump: \n{self.__dict__.items()}\n")


class UserPrompt(IndividualPrompt):
    name: str = ""

    def model_post_init(self, __context):
        logger.info(f"UserPrompt instance created: {self}")
        self.prompt.prompt_type = PromptType.USER  # "user"


class SystemPrompt(IndividualPrompt):
    name: str = ""

    def model_post_init(self, __context):
        logger.info(f"SystemPrompt instance created: {self}")
        self.prompt.prompt_type = PromptType.SYSTEM  # "system"


class PromptContainer(BaseModel):
    prompts: List[IndividualPrompt]

    _prompt_dict: Dict[str, PromptInstance] = PrivateAttr(
        default_factory=dict
    )  # = Field(default_factory=dict)

    @field_validator("prompts")
    def check_prompt_names(cls, prompts):
        # logger.info(f"Checking prompt names: {prompts}")
        for prompt in prompts:
            if not prompt.name:
                raise ValueError("Each IndividualPrompt must have a non-empty 'name'.")
        return prompts

    def __init__(self, **data):
        super().__init__(**data)
        self._create_prompt_dict()

    def _create_prompt_dict(self):
        # logger.info(f"Creating prompt dict: {self.prompts}")
        prompt_dict = {prompt.name: prompt.prompt for prompt in self.prompts}
        # logger.info(f"Prompt dict created: \n\n{prompt_dict}\n")
        self.__dict__["_prompt_dict"] = prompt_dict
        # self._prompt_dict = prompt_dict
        # logger.info(f"\n\nPrompt dict created: {self._prompt_dict}\n")

    def __getattr__(self, name):
        # prompt_dict = self.__dict__.get("_prompt_dict", {})
        prompt_dict = self._prompt_dict
        # logger.info(f"Getting attribute: {name}\nPrompt Dict: {prompt_dict}\n")
        if name in prompt_dict:
            prompt_instance = prompt_dict[name]
            # logger.info(
            #    f"Prompt instance | Type: {type(prompt_instance)}\n\n{prompt_instance}\n"
            # )
            return prompt_instance
        raise AttributeError(
            f"\n\n'{self.__class__.__name__}' object has no attribute '{name}'\n"
        )


class PromptModifiers(PromptContainer): ...


class PromptGroup(PromptContainer): ...


class PromptBase(BaseModel):
    user_prompt: str = ""
    system_prompt: Optional[str] = None
    version: str = "0.0.0"

    # Computed Fields:
    @computed_field
    @property
    def prompt(self) -> str:
        return self.build_prompt()

    @computed_field
    @property
    def input_attrs(self) -> List[str]:
        return self.extract_input_attrs() or []

    @computed_field
    @property
    def metadata(self) -> dict:
        metadata = {
            "tags": {f"version:{self.version}", f"category:{self.__class__.__name__}"}
        }
        return metadata

    # Functions:
    def build_prompt(self) -> str:
        built_prompt = (
            system_header + self.system_prompt + "\n\n" + user_header + self.user_prompt
            if self.system_prompt
            else self.user_prompt
        )
        return built_prompt

    def extract_input_attrs(self) -> List[str]:
        attr_list = extract_bracketed_text(self.prompt)
        return attr_list


class PromptModel(PromptBase):
    user_prompts: Optional[PromptContainer]  # List[IndividualPrompt] = []
    system_prompts: Optional[PromptContainer] = None

    # @field_validator("user_prompts", "system_prompts", mode="before")
    # @classmethod
    # def validate_prompts(cls, val) -> PromptContainer:
    #    logger.info(
    #        f"Validating Prompts | Is PromptContainer subclass: {isinstance(val,PromptContainer)} | Value: {val}"
    #    )
    #    logger.info(f"Type: {type(val)}")
    #    if isinstance(val, PromptContainer):
    #        return val
    #    raise TypeError(
    #        "Wrong type for 'User_prompts', must be a subclass of 'PromptContainer'"
    #    )

    def select_prompts(
        self, user_prompt_name: str, system_prompt_name: Optional[str] = None
    ):
        # Find and set user prompt
        user_prompt = next(
            (p for p in self.user_prompts if p.name == user_prompt_name), None
        )
        if not user_prompt:
            raise ValueError(f"User prompt '{user_prompt_name}' not found")
        self.selected_user_prompt = user_prompt_name
        self.user_prompt = user_prompt.prompt

        # Find and set system prompt if provided
        if system_prompt_name:
            if not self.system_prompts:
                raise ValueError("No system prompts available")
            system_prompt = next(
                (p for p in self.system_prompts if p.name == system_prompt_name), None
            )
            if not system_prompt:
                raise ValueError(f"System prompt '{system_prompt_name}' not found")
            self.selected_system_prompt = system_prompt_name
            self.system_prompt = system_prompt.prompt
        else:
            self.selected_system_prompt = None
            self.system_prompt = None

        return self.prompt

    def __call__(self, user_prompt_name: str, system_prompt_name: Optional[str] = None):
        return self.select_prompts(user_prompt_name, system_prompt_name)
