from datetime import datetime
from typing import Optional, Dict, Type, Any, ClassVar, Iterator
from pydantic import BaseModel, PrivateAttr
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    MetaData,
    Table,
    insert,
)
from sqlmodel import (
    SQLModel,
    create_engine as sqlmodel_create_engine,
    # Session,
    select,
    # Column as SQLColumn,
    # JSONB,
    # sa_column,
    # Field as SQLField,
    # MetaData as SQLMetaData,
)

from sqlalchemy.orm import sessionmaker
import uuid
import json
from contextlib import contextmanager
from contextvars import ContextVar


# Import init vars:
from miragram.src.base.config import db_url

print(f"\n\ndb_url:\n\n{db_url}\n\n")


# Configure logging
from miragram.log.logger import get_logger

logger = get_logger("MiraBase")


# Functions ---------------------------------------------------------------------------------------------------------


_init_context_var = ContextVar("_init_context_var", default=None)


@contextmanager
def init_context(value: Dict[str, Any]) -> Iterator[None]:
    token = _init_context_var.set(value)
    try:
        yield
    finally:
        _init_context_var.reset(token)


def reconstruct_instance_from_db(class_name: str, json_data: dict):
    target_cls = MiraResponse.class_registry.get(class_name)
    if not target_cls:
        raise ValueError(f"Class '{class_name}' not found in registry.")
    logger.info(f"Base | Reconstructing class from json: \n\n{json_data}\n\n")
    parsed_obj = target_cls.from_json_data(json_data)
    return parsed_obj


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


# Classes -----------------------------------------------------------------------------------------------------------


# Singleton instance initialization:
class SingletonEngine:
    _instance = None
    _sqlmodel_instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = create_engine(db_url)
        return cls._instance

    @classmethod
    def get_sqlmodel_instance(cls):
        if cls._sqlmodel_instance is None:
            cls._sqlmodel_instance = sqlmodel_create_engine(db_url)
            SQLModel.metadata.create_all(cls._sqlmodel_instance)
        return cls._sqlmodel_instance


# MiraBase class
class MiraBase(BaseModel):
    id: Optional[str] = None
    _tables: ClassVar[Dict[str, Any]] = {}
    class_registry: ClassVar[Dict[str, Type["MiraBase"]]] = {}

    _new: bool = PrivateAttr(default=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.id = data.get("id") or uuid.uuid4().hex
        context = _init_context_var.get()
        if context and "new" in context:
            self._new = context["new"]
        else:
            self._new = True

        if self._new and self.__class__.__name__.lower()[0:7] != "partial":
            ## Set an ID for the instance
            self.id = uuid.uuid4().hex
            logger.info(f"SAVING INSTANCE: {self.__class__.__name__.lower()} {self}")
            self.save()
        else:
            self.id = data["id"]

    def save(self):
        json_data = json.loads(self.model_dump_json())
        class_name = self.__class__.__name__.lower()

        engine = SingletonEngine.get_instance()
        Session = sessionmaker(bind=engine)
        session = Session()

        # Use cached table if available
        if class_name in self._tables:
            table = self._tables[class_name]
        else:
            metadata = MetaData()
            metadata.reflect(bind=engine)
            if class_name not in metadata.tables:
                table = Table(
                    class_name,
                    metadata,
                    Column("id", String, primary_key=True),
                    Column("json_data", JSON),
                )
                metadata.create_all(engine)
            else:
                table = metadata.tables[class_name]
            self._tables[class_name] = table  # Cache the table

        # Insert the JSON data into the table
        try:
            insert_stmt = insert(table).values(id=self.id, json_data=json_data)
            session.execute(insert_stmt)
            session.commit()
            logger.info(f"Saved {self.__class__.__name__} with ID {self.id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving {self.__class__.__name__}: {e}")
            raise e
        finally:
            session.close()

    @classmethod
    def from_json_data(cls, json_data):
        target_cls = cls.class_registry.get(cls.__name__)
        if not target_cls:
            raise ValueError(f"Class '{cls.__name__}' not found in registry.")
        logger.info(f"Creating {cls.__name__} from JSON data")
        instance = target_cls.parse_obj(json_data)
        # instance._new = False
        with init_context({"new": False}):
            instance = target_cls.parse_obj(json_data)

        return instance

    @classmethod
    def create(cls, **data):
        instance = cls(**data)
        instance.save()
        return instance

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.class_registry[cls.__name__] = cls


# MiraResponse class
class MiraResponse(MiraBase):
    content: Optional[str] = None

    # Additional methods or overrides if necessary


"""
# MiraCall class
class MiraCall(MiraBase):
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    response_model: Optional[Type[BaseModel]] = None
    json_mode: bool = True

    def __init__(
        self,
        system_prompt=None,
        prompt_template=None,
        response_model=None,
        json_mode=True,
        **data,
    ):
        super().__init__(
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            response_model=response_model,
            json_mode=json_mode,
            **data,
        )
        if not self.call_time:
            self.call_time = datetime.now()
        if not self.chat_id:
            self.chat_id = data.get("chat_id") or uuid.uuid4().hex

    def __call__(self, func):
        # Store the options in the function's attributes
        func._system_prompt = self.system_prompt
        func._prompt_template = self.prompt_template
        func._response_model = self.response_model
        func._json_mode = self.json_mode
        return func
"""


"""
# MiraCall class
class MiraCall(MiraBase):
    chat_id: Optional[str] = None
    call_time: Optional[datetime] = None
    input: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.call_time:
            self.call_time = datetime.now()
        if not self.chat_id:
            self.chat_id = data.get("chat_id") or None


"""


# MiraChat class
class MiraChat(MiraBase):
    chat_id: str
    chat_type: str

    _engine: Any = PrivateAttr()
    _chat_table: Any = PrivateAttr()

    def __init__(
        self, chat_id: Optional[str] = None, chat_type: Optional[str] = None, **data
    ):
        super().__init__(**data)
        self.chat_id = chat_id or uuid.uuid4().hex
        self.chat_type = chat_type or self.__class__.__name__
        self._engine = SingletonEngine.get_instance()
        self._ensure_chat_table()

    def _ensure_chat_table(self):
        metadata = MetaData()
        metadata.reflect(bind=self._engine)
        if "chat" not in metadata.tables:
            self._chat_table = Table(
                "chat",
                metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("chat_id", String),
                Column("chat_type", String),
                Column("io_instance_type", String),
                Column("io_instance_id", String),
                Column("timestamp", DateTime),
            )
            metadata.create_all(self._engine)
        else:
            self._chat_table = metadata.tables["chat"]

    def send_call(self, input_text: str) -> MiraResponse:
        # Create and save MiraCall instance
        call = MiraCall(chat_id=self.chat_id, input=input_text)
        self._save_chat_record("call", call.id)

        # Send to LLM API and get response (Placeholder)
        response_data = self.send_to_llm_api(input_text)

        # Create and save MiraResponse instance
        response = MiraResponse(**response_data)
        self._save_chat_record("response", response.id)

        return response

    def send_to_llm_api(self, input_text: str) -> dict:
        # Placeholder for actual LLM API call
        logger.info(f"Sending to LLM API: {input_text}")
        # Simulate LLM response
        response_data = {"content": f"Response to '{input_text}'"}
        return response_data

    def _save_chat_record(self, io_instance_type: str, io_instance_id: str):
        session = sessionmaker(bind=self._engine)()
        timestamp = datetime.now()
        chat_record = {
            "chat_id": self.chat_id,
            "chat_type": self.chat_type,
            "io_instance_type": io_instance_type,
            "io_instance_id": io_instance_id,
            "timestamp": timestamp,
        }
        try:
            insert_stmt = insert(self._chat_table).values(**chat_record)
            session.execute(insert_stmt)
            session.commit()
            logger.info(f"Saved chat record: {chat_record}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save chat record: {e}")
            raise e
        finally:
            session.close()
