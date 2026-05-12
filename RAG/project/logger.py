import logging
import sys


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GREY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Colors.GREY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED,
    }

    def format(self, record: logging.LogRecord):
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        message = super().format(record)
        return f"{color}{message}{Colors.RESET}"


class Logger:
    def __init__(self, log_file: str = "app.log", color_file: bool = False) -> None:
        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Format
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        # ---- Console Handler (Colored) ----
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter(fmt, datefmt))
        self.logger.addHandler(console_handler)

        # ---- File Handler ----
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        if color_file:
            file_handler.setFormatter(ColoredFormatter(fmt, datefmt))
        else:
            file_handler.setFormatter(logging.Formatter(fmt, datefmt))

        self.logger.addHandler(file_handler)

    # ---- Methods ----
    def header(self, message: str):
        line = "=" * 60
        purple_line = f"{Colors.PURPLE}{line}{Colors.RESET}"
        purple_msg = f"{Colors.PURPLE}{message}{Colors.RESET}"

        self.logger.info(purple_line)
        self.logger.info(purple_msg)
        self.logger.info(purple_line)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def debug(self, message: str):
        self.logger.debug(message)