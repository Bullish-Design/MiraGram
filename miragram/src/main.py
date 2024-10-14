# Imports


# Local Imports
from miragram.src.base import (
    SingletonEngine,
    MiraBase,
    MiraCall,
    MiraResponse,
    MiraChat,
)


# Logging
from miragram.log.logger import get_logger

logger = get_logger("MiraGram")


def main():
    logger.info("Starting MiraGram")


if __name__ == "__main__":
    main()
