import logging
import logging.config
import os


# Define log file location
from constants import PY_PROJECT_ROOT
LOG_DIR = os.path.join(PY_PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(PY_PROJECT_ROOT, LOG_DIR, "app.log")

# Init log dir and file
os.makedirs(LOG_DIR, exist_ok=True)
with open(LOG_FILE, "w") as f:
    f.write("Hello, World!")


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "<%(asctime)s> <%(levelname)-8s> <%(module)s>: %(message)s"
        },
        "simple": {
            "format": "<%(levelname)-8s>: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "formatter": "detailed",
            "level": "DEBUG"
        }
    },
    "root": {  # Applies to all loggers
        "handlers": ["console", "file"],
        "level": "DEBUG"
    }
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Create a logger for modules to use
def get_logger(name):
    return logging.getLogger(name)

# L = logging.getLogger(__name__)

# Test different log levels
# L.debug("This is a DEBUG message")
# L.info("This is an INFO message")
# L.warning("This is a WARNING message")
# L.error("This is an ERROR message")
# L.critical("This is a CRITICAL message")