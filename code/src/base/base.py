from datetime import datetime
from typing import Optional, Dict, Type, Any, ClassVar
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
from sqlalchemy.orm import sessionmaker
import uuid
import json
import logging

# Configure logging
from miragram.log.logger import get_logger

logger = get_logger("MiraBase")


# SingletonEngine placeholder (replace with your actual implementation)
class SingletonEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = create_engine("sqlite:///mira.db")  # Example with SQLite
        return cls._instance


# MiraBase class
class MiraBase(BaseModel):
    id: Optional[str] = None
    _tables: ClassVar[Dict[str, Any]] = {}
    class_registry: ClassVar[Dict[str, Type["MiraBase"]]] = {}

    _new: bool = PrivateAttr(default=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.id = data.get("id") or uuid.uuid4().hex
        self._new = data.get("_new", True)
        if self._new:
            logger.info(f"Saving instance of {self.__class__.__name__}: {self}")
            self.save()

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
        instance._new = False
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
            self.chat_id = data.get("chat_id") or uuid.uuid4().hex


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
