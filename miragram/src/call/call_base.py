# Imports ---------------------------------------------------------------------------
import uuid
from datetime import datetime
import functools

from typing import (
    Any,
    Optional,
    Type,
    List,
    Iterator,
    Union,
    Type,
)

# Pydantic Imports ------------------------------------------------------------------
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    field_serializer,
    InstanceOf,
)

# SQLModel Imports ------------------------------------------------------------------
from sqlmodel import (
    SQLModel,
    #    create_engine as sqlmodel_create_engine,
    Session,
    select,
    Column as SQLColumn,
    # JSONB,
    # sa_column,
    Field as SQLField,
    #    MetaData as SQLMetaData,
)


# SQLAlchemy Imports ----------------------------------------------------------------
from sqlalchemy import (
    MetaData,
    #    Table,
    Column,
    Integer,
    #    DateTime,
    #    Text,
    #    String,
    #    create_engine,
    JSON,
    #    DateTime,
    #    insert,
    inspect as sql_inspect,
)

from sqlalchemy.orm import sessionmaker, declarative_base


# Mirascope Imports -----------------------------------------------------------------
from openai import OpenAI

from mirascope.core import (
    openai,
    prompt_template,
    metadata,
)

# from mirascope.core.base.stream import BaseStream  #
from mirascope.core.base.structured_stream import BaseStructuredStream


# Library Imports -------------------------------------------------------------------
from miragram.src.base.base import (
    SingletonEngine,
    MiraResponse,
    MiraBase,
    #    MiraCall,
    #    MiraChat,
)
from miragram.src.base.config import (
    local_model_url,
    local_model_name,
)

# Tenacity Imports ------------------------------------------------------------------
from tenacity import retry, stop_after_attempt, wait_exponential


# Configure logging
from miragram.log.logger import get_logger

logger = get_logger("CallBase")

client = OpenAI(
    api_key="ollama",
    base_url=local_model_url,
)


# Functions -------------------------------------------------------------------------
# Functions to reconstruct instances
def reconstruct_instance_from_db(class_name: str, json_data: dict):
    target_cls = MiraResponse.class_registry.get(class_name)
    if not target_cls:
        raise ValueError(f"Class '{class_name}' not found in registry.")
    logger.info(f"Base | Reconstructing class from json: \n\n{json_data}\n\n")
    parsed_obj = target_cls.from_json_data(json_data)
    return parsed_obj


def get_instances_from_db(class_name: str):
    engine = SingletonEngine.get_instance()
    Session = sessionmaker(bind=engine)
    session = Session()

    table_name = class_name.lower()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables.get(table_name)
    if table is None:
        print(f"Table '{table_name}' does not exist in the database.")
        return []
    # Fetch all records
    stmt = select(table)
    results = session.execute(stmt).fetchall()
    instances = []
    for row in results:
        # print(f"\nTable Row type: {type(row[1])}\n\n{row[1]}\n")
        json_data = row[1]
        instance = reconstruct_instance_from_db(class_name, json_data)
        instances.append(instance)

    session.close()
    return instances


def get_single_instance_from_db(class_name: str, instance_id: str):
    engine = SingletonEngine.get_instance()
    Session = sessionmaker(bind=engine)
    session = Session()

    table_name = class_name.lower()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables.get(table_name)

    if table is None:
        raise ValueError(f"Table '{table_name}' does not exist in the database.")

    try:
        # Fetch the record with the matching ID
        stmt = select(table).where(table.c.json_data["id"].astext == instance_id)
        result = session.execute(stmt).one()
        json_data = result[1]  # Assuming the JSON data is in the second column
        instance = reconstruct_instance_from_db(class_name, json_data)
        return instance  # s

    except NoResultFound:
        raise ValueError(
            f"No instance found with id '{instance_id}' in table '{table_name}'."
        )
    finally:
        session.close()


# Classes --------------------------------------------------------------------------
# Intermediate Steps:
class AskStep(BaseModel):
    response_type: str
    response_id: str


class AskResult(BaseModel):
    query: str
    step_response_list: List[AskStep]


# Dynamic database access model:
class DynamicModel:
    def __init__(self, table_name: str):  # , engine: Engine):
        self.table_name = table_name
        self.engine = SingletonEngine.get_sqlmodel_instance()
        self.model = self._create_dynamic_model(table_name)

    def _create_dynamic_model(self, table_name: str) -> Type[SQLModel]:
        # Define a dynamic SQLModel class with the given table name
        class DynamicSQLModel(SQLModel, table=True):
            __tablename__ = table_name
            id: Optional[int] = SQLField(
                sa_column=SQLColumn(Integer, primary_key=True, autoincrement=True)
            )
            json_data: dict = SQLField(sa_column=Column(JSON))
            created: Optional[datetime] = SQLField(default_factory=datetime.now)
            cost: Optional[float] = SQLField(default=None)

        return DynamicSQLModel

    def get_all_rows(self):
        with Session(self.engine) as session:
            # Check if the table exists
            inspector = sql_inspect(self.engine)
            if self.table_name not in inspector.get_table_names():
                print(f"Table '{self.table_name}' does not exist in the database.")
                return []

            # Query all rows
            results = session.query(self.model).all()
            return results

    def reconstruct_instance_from_db(self, class_name: str, json_data: dict):
        target_cls = MiraResponse._class_registry.get(class_name)
        if not target_cls:
            raise ValueError(f"Class '{class_name}' not found in registry.")
        return target_cls.parse_obj(json_data)

    def search_in_json_data(self, key: str, value):
        with Session(self.engine) as session:
            stmt = session.query(self.model).filter(
                self.model.json_data[key].astext == str(value)
            )
            results = stmt.all()
            for instance in results:
                print(f"id: {instance.id}")
                print(f"json_data: {instance.json_data}")
                print()


# """
class IntermediateResponse(SQLModel, table=True):
    # id: Optional[int] = SQLField(default=None, primary_key=True)
    id: uuid.UUID = SQLField(default_factory=uuid.uuid4, primary_key=True)
    # call_id: Optional[str] = SQLField(default=None)
    function: str
    llm_model: str
    timestamp: datetime
    input: Optional[dict] = SQLField(sa_column=Column(JSON))  # Store as JSON
    output: Optional[dict] = SQLField(sa_column=Column(JSON))
    prompt_template: str
    meta: Optional[dict] = SQLField(default=None, sa_column=Column(JSON))
    successful: Optional[bool] = None
    cost: Optional[float] = SQLField(default=None)
    # TODO: Update to store the cost of each call


# """


# Generic decorator class
class Decorator:
    def __init__(self, decorator_func, *args, **kwargs):
        """
        Generic class for any decorator.
        :param decorator_func: The decorator function to apply.
        :param args: Positional arguments for the decorator function.
        :param kwargs: Keyword arguments for the decorator function.
        """
        self.decorator_func = decorator_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func):
        """Applies the stored decorator function with logging of inputs and outputs."""

        # Apply the original decorator
        decorated_func = self.decorator_func(*self.args, **self.kwargs)(func)

        @functools.wraps(decorated_func)
        def wrapper(*func_args, **func_kwargs):
            result = decorated_func(*func_args, **func_kwargs)
            return result

        return wrapper


class RetryDecorator(Decorator):
    def __init__(self, stop_func, wait_func):
        retry_func = functools.partial(retry, stop=stop_func, wait=wait_func)
        super().__init__(retry_func)


# Combined decorator class
class CombinedDecorator:
    def __init__(self, *decorators):
        self.decorators = decorators

    def __call__(self, func):
        for decorator in reversed(self.decorators):
            func = decorator(func)
        return func


# Generic combined decorator class
class CompleteDecorator(CombinedDecorator):
    def __init__(self, *decorators):
        """
        Combines multiple decorators and stores inputs/outputs in a Postgres DB.
        :param decorators: A list of Decorator instances.
        # TODO: Handle validation errors, and store the partial results in the DB
        """
        self.decorators = decorators
        self.llm_model = None  # To store model type from openai.call
        self.prompt_template = None  # To store prompt template
        self.meta = None  # To store metadata
        self.intermediate_response = None
        self.stream = False
        self.response_model = None

    def store_response(
        self,
        func_name,
        llm_model,
        input_data,
        output_data,
        prompt_template,
        metadata,
        cost,
        # created,
    ):
        """Store the function call input and output into the database."""
        # BOOOOOOOO - Just make this be the only thing that stores the result?
        # TODO:  Determine which way to go for the final implementation
        #             - Add an error handling mechanism
        # if isinstance(output_data, MiraResponse):
        #    output_data.save()
        # output_data = MiraResponse.model_validate(output_data)
        # logger.info(f"Combined Decorator | MiraResponse Saved.")

        if isinstance(output_data, BaseModel):
            output_data = output_data.model_dump()
        response = IntermediateResponse(
            function=func_name,
            llm_model=llm_model,
            timestamp=datetime.now(),
            # timestamp=output_data["created"],
            input=input_data,
            output=output_data,
            prompt_template=prompt_template,
            cost=cost,
            # meta=metadata,
        )

        self.intermediate_response = response
        engine = SingletonEngine.get_sqlmodel_instance()

        # Store the response in the database
        with Session(engine) as session:
            session.add(response)
            session.commit()

    def extract_model_type(self):
        """Extracts the model type from the openai.call decorator if present."""
        for decorator in self.decorators:
            if decorator.decorator_func == openai.call:
                self.llm_model = decorator.kwargs.get("model")
                self.stream = decorator.kwargs.get("stream", False)
                break

    def extract_response_model(self):
        """Extracts the response model from the openai.call decorator if present."""
        for decorator in self.decorators:
            if decorator.decorator_func == openai.call:
                self.response_model = decorator.kwargs.get("response_model")
                break

    def extract_prompt_template(self):
        """Extracts the prompt template from the prompt_template decorator if present."""
        for decorator in self.decorators:
            if decorator.decorator_func == prompt_template:
                self.prompt_template = decorator.args[0]
                #
                break

    def extract_metadata(self):
        """Extracts the metadata from the metadata decorator if present."""
        for decorator in self.decorators:
            if decorator.decorator_func == metadata:
                self.meta = decorator.args[0]
                break

    def __call__(self, func):
        """Applies all decorators in reverse order and stores results."""
        self.extract_model_type()  # Extract model type before applying decorators
        self.extract_prompt_template()  # Extract prompt template before applying decorators
        self.extract_metadata()  # Extract metadata before applying decorators
        self.extract_response_model()

        for decorator in reversed(self.decorators):
            func = decorator(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            input_data = f"args: {args}, kwargs: {kwargs}"

            # Execute the function
            try:
                result = func(*args, **kwargs)
            except ValidationError as e:
                logger.error(f"Validation Error: {e}")

            model_type = self.llm_model if self.llm_model else "Unknown"
            prompt_template = (
                self.prompt_template if self.prompt_template else "Unknown"
            )
            metadata = self.meta if self.meta else {}

            if self.stream:
                if self.response_model is not None:
                    # Streaming with response_model
                    return self.handle_streaming_response_with_response_model(
                        result,
                        func.__name__,
                        model_type,
                        input_data,
                        prompt_template,
                        metadata,
                    )
                else:
                    # Streaming without response_model
                    return self.handle_streaming_response(
                        result,
                        func.__name__,
                        model_type,
                        input_data,
                        prompt_template,
                        metadata,
                    )
            else:
                # Non-streaming response handling
                return self.handle_non_streaming_response(
                    result,
                    func.__name__,
                    model_type,
                    input_data,
                    prompt_template,
                    metadata,
                )

            """
            if isinstance(result, BaseStream):
                # Streaming response
                logger.info(f"Streaming Response: {type(result)}: \n\n{result}\n\n")
                output_data = []
                for message in result:
                    output_data.append(message)
                cost = None  # Adjust if cost information is available
            else:
                logger.info(f"Non-Streaming Response: {type(result)}: \n\n{result}\n\n")
                # Non-streaming response
                output_data = result
                cost = None  # result._response.cost
            logger.info(f"Cost: {cost}\n\n\n")
            # Store the input/output in the database
            self.store_response(
                func_name=func.__name__,
                llm_model=model_type,
                input_data=input_data,
                output_data=result,
                prompt_template=prompt_template,
                metadata=metadata,
                cost=cost,
            )
            """
            return result

        return wrapper

    def handle_streaming_response(
        self,
        result: BaseStructuredStream,
        func_name: str,
        llm_model: str,
        input_data: Any,
        prompt_template: str,
        metadata: dict,
    ) -> Iterator:
        """Handles streaming LLM responses."""
        collected_chunks = []

        def generator():
            try:
                for chunk, tool in result.stream:
                    collected_chunks.append(chunk)
                    yield chunk, tool
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                raise e
            finally:
                # After iteration is complete, store the response
                output_data = [chunk.content for chunk in collected_chunks]
                cost = result.stream.cost  # If available
                self.store_response(
                    func_name=func_name,
                    llm_model=llm_model,
                    input_data=input_data,
                    output_data=output_data,
                    prompt_template=prompt_template,
                    metadata=metadata,
                    cost=cost,
                )

        return generator()

    def handle_streaming_response_with_response_model(
        self,
        result: BaseStructuredStream,
        func_name: str,
        llm_model: str,
        input_data: Any,
        prompt_template: str,
        metadata: dict,
    ) -> Iterator:
        """Handles streaming LLM responses when response_model is set."""
        collected_models = []

        def generator():
            # logger.info(f"Response model Result: {result}")
            try:
                for partial_model in result:
                    # logger.info(f"Yielding partial model: {partial_model}")
                    collected_models.append(partial_model)
                    yield partial_model
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                raise e
            else:
                # After iteration is complete, store the response
                # finally:
                final_model = result.constructed_response_model
                logger.info(f"Final model: {final_model}")
                cost = getattr(result.stream, "cost", None)
                self.store_response(
                    func_name=func_name,
                    llm_model=llm_model,
                    input_data=input_data,
                    output_data=final_model.model_dump(),
                    prompt_template=prompt_template,
                    metadata=metadata,
                    cost=cost,
                )

        return generator()

    def handle_non_streaming_response(
        self,
        result: Any,
        func_name: str,
        llm_model: str,
        input_data: Any,
        prompt_template: str,
        metadata: dict,
    ) -> Any:
        """Handles non-streaming LLM responses."""
        output_data = result
        cost = (
            output_data._response.cost
        )  # getattr(result, "_response", {}).get("cost", None)
        logger.info(f"Cost: {cost}\n\n\n")
        # Store the input/output in the database
        self.store_response(
            func_name=func_name,
            llm_model=llm_model,
            input_data=input_data,
            output_data=output_data,
            prompt_template=prompt_template,
            metadata=metadata,
            cost=cost,
        )
        return result


# """
# Old LLM_Call class
class LLM_Call:
    def __init__(
        self,
        prompt_template=None,
        response_model=None,
        version=None,
        category=None,
        model="gpt-4o-mini",
        system_prompt=None,
        json_mode=True,
        retries=3,
        retry_min=1,
        retry_max=5,
        stream=False,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.version = version
        self.category = category
        self.meta = None
        self.response_model = response_model
        self.json_mode = json_mode
        self.retries = retries
        self.retry_min = retry_min
        self.retry_max = retry_max
        self.stream = stream

    def generate_call_decorator(self):
        call_decorator = Decorator(
            openai.call,
            model=self.model,
            response_model=self.response_model,
            json_mode=self.json_mode,
            stream=self.stream,
        )
        return call_decorator

    def generate_prompt_decorator(self):
        if self.system_prompt:
            prompt = self.system_prompt + "\n\n" + self.prompt_template
        else:
            prompt = self.prompt_template
        prompt_decorator = Decorator(prompt_template, prompt)
        return prompt_decorator

    def generate_metadata_decorator(self):
        self.meta = {"tags": {f"version:{self.version}", f"category:{self.category}"}}
        metadata_decorator = Decorator(metadata, self.meta)
        return metadata_decorator

    def generate_retry_decorator(self):
        retry_decorator = RetryDecorator(
            stop_func=stop_after_attempt(self.retries),
            wait_func=wait_exponential(
                multiplier=1, min=self.retry_min, max=self.retry_max
            ),
        )
        return retry_decorator

    def generate_combined_decorator(self):
        call_decorator = self.generate_call_decorator()
        prompt_decorator = self.generate_prompt_decorator()
        metadata_decorator = self.generate_metadata_decorator()
        retry_decorator = self.generate_retry_decorator()

        combined_decorator = CombinedDecorator(
            retry_decorator,
            call_decorator,
            prompt_decorator,
            metadata_decorator,
        )
        return combined_decorator

    def __call__(self, func):
        combined_decorator = self.generate_combined_decorator()
        # if self.stream:
        #    result = combined_decorator(func)
        #    yield result
        # else:
        result = combined_decorator(func)
        return result


# """
def all_subclasses(cls):
    subclasses = list(cls.__subclasses__()) + [
        s for c in cls.__subclasses__() for s in all_subclasses(c)
    ]
    logger.info(f"Subclasses: {subclasses}")
    return subclasses


def model_instance(cls):
    instances = Union.__getitem__(tuple(all_subclasses(cls)[::-1]))
    logger.info(f"Model Instances: {instances}")
    return instances


# Updated MiraCall class:
# Ideally auto saves the call params to the database
# Should serve as a decorator for any call functions, and have additional options passed into it with an extra LLM_Call_Options decorator on top of it.
class MiraCall(MiraBase):
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    response_model: Optional[Type[MiraResponse]] = None  # | list[MiraResponse]] = None)
    llm_model: Optional[str] = "gpt-4o-mini"
    json_mode: bool = True
    chat_id: Optional[str] = None
    call_time: Optional[datetime] = None
    stream: Optional[bool] = False
    local: Optional[bool] = False

    """
    def __init__(
        self,
        system_prompt=None,
        prompt_template=None,
        response_model=None,
        json_mode=True,
        **data,
    ):
        super().__init__(**data)
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.response_model = response_model
        self.json_mode = json_mode

        if not self.call_time:
            self.call_time = datetime.now()
        if not self.chat_id:
            self.chat_id = data.get("chat_id") or None
    """

    @field_validator("response_model", mode="before")
    @classmethod
    def validate_response_model(cls, val):
        logger.info(
            f"Validating Response Model | Is MiraResponse subclass: {issubclass(val, MiraResponse)} | Value: {val}"
        )
        if val is None or (isinstance(val, type) and issubclass(val, MiraResponse)):
            return val
        raise TypeError(
            "Wrong type for 'response_model', must be a subclass of MiraResponse"
        )

    @field_serializer("response_model")
    def serialize_response_model(self, response_model):
        if response_model is None:
            return None
        return_response_model = (
            response_model.__module__ + "." + response_model.__qualname__
        )
        logger.info(f"Serialized Response Model: {return_response_model}")
        return return_response_model

    def call_decorator(self):
        if self.local:
            openai_call = openai.call
            # openai_call = openai.call(model="llama", client=client)
            call_decorator = Decorator(
                openai_call,
                model=local_model_name,
                client=client,
                response_model=self.response_model,
                json_mode=self.json_mode,
                stream=self.stream,
            )
        else:
            openai_call = openai.call
            # openai_call = openai.call
            call_decorator = Decorator(
                openai_call,
                model=self.llm_model,
                response_model=self.response_model,
                json_mode=self.json_mode,
                stream=self.stream,
            )
        return call_decorator

    def prompt_decorator(self):
        if self.system_prompt:
            prompt = self.system_prompt + "\n\n" + self.prompt_template
        else:
            prompt = self.prompt_template
        prompt_decorator = Decorator(prompt_template, prompt)
        return prompt_decorator

    def generate_decorator(self):
        call_decorator = self.call_decorator()
        prompt_decorator = self.prompt_decorator()
        combined_decorator = CombinedDecorator(call_decorator, prompt_decorator)
        return combined_decorator

    def __call__(self, func):
        # Store the options in the function's attributes
        logger.info(f"MiraCall | __call__ | {func.__name__}")
        func._system_prompt = self.system_prompt
        func._prompt_template = self.prompt_template
        func._response_model = self.response_model
        func._json_mode = self.json_mode
        miracall_decorator = self.generate_decorator()
        return miracall_decorator(func)
        # return func


# Metadata options class
class MetadataOptions:
    def __init__(self, version=None, category=None):
        self.version = version
        self.category = category

    def get_metadata(self):
        return {"tags": {f"version:{self.version}", f"category:{self.category}"}}

    def decorator(self):
        return Decorator(metadata, self.get_metadata())


# LLM model options class
class LLMModelOptions:
    def __init__(self, model="gpt-4o-mini", stream=False):
        self.model = model
        self.stream = stream

    def get_model_options(self):
        return {"model": self.model, "stream": self.stream}


# Retry options class
class RetryOptions:
    def __init__(self, retries=3, retry_min=1, retry_max=5):
        self.retries = retries
        self.retry_min = retry_min
        self.retry_max = retry_max

    def decorator(self):
        return RetryDecorator(
            stop_func=stop_after_attempt(self.retries),
            wait_func=wait_exponential(
                multiplier=1, min=self.retry_min, max=self.retry_max
            ),
        )


# New LLM_Call class
class LLM_Call_Options:  # (BaseModel):
    # metadata_options: Optional[MetadataOptions] = None
    # llm_model_options: Optional[LLMModelOptions] = None
    # retry_options: Optional[RetryOptions] = None

    def __init__(
        self,
        metadata_options=None,
        # llm_model_options=None,
        retry_options=None,
    ):
        self.metadata_options = metadata_options or None  # MetadataOptions()
        # self.llm_model_options = llm_model_options or LLMModelOptions()
        self.retry_options = retry_options or RetryOptions()

    def __call__(self, func):
        # Extract prompt options from the function's attributes
        logger.info(f"LLM_Call_Options | __call__ | {func.__name__}")
        system_prompt = getattr(func, "_system_prompt", None)
        prompt_template_str = getattr(func, "_prompt_template", "")
        response_model = getattr(func, "_response_model", None)
        json_mode = getattr(func, "_json_mode", True)
        call_model = getattr(func, "_llm_model", "gpt-4o-mini")
        stream = getattr(func, "_stream", False)

        # Build the prompt
        if system_prompt:
            prompt = system_prompt + "\n\n" + prompt_template_str
        else:
            prompt = prompt_template_str

        # Create decorators
        prompt_decorator = Decorator(prompt_template, prompt)

        if self.metadata_options:
            metadata_decorator = Decorator(
                metadata, self.metadata_options.get_metadata()
            )

        # model_options = self.llm_model_options.get_model_options()
        call_decorator = Decorator(
            openai.call,
            model=call_model,
            response_model=response_model,
            json_mode=json_mode,
            stream=stream,
        )

        retry_decorator = self.retry_options.get_retry_decorator()

        if self.metadata_options:
            combined_decorator = CombinedDecorator(
                retry_decorator,
                call_decorator,
                prompt_decorator,
                metadata_decorator,
            )
        else:
            combined_decorator = CombinedDecorator(
                retry_decorator,
                call_decorator,
                prompt_decorator,
            )

        return combined_decorator(func)


# Misc ------------------------------------------------------------------------------
