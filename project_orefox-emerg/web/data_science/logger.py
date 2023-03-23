"""Holds a basic interface with the Python logger for the other files."""

import logging
from django.conf import settings
import os

class Logger:
    """Class to hold logging of other modules functionality.
    
    Attributes:
        logger: Instance of logging.Logger to handle logging.
    """

    def __init__(self, logger_name: str, level: str='I'):
        """Defines a logger to be used by the other modules.
        
        Args:
            logger_name: The name for the logger. Two different loggers called
              with the same name will refer to the same logger.
            level: Character to set the logger to only log with specific
              messages. 'I' logs the least (only info), 'W' will also log
              warnings, and all other inputs will log every message.
        """

        # Get the logger, this will get the same one if the names are the same
        self.logger = logging.getLogger(logger_name)
        
        # Set the logger to only record at the appropriate level
        if level == 'I':
            self.logger.setLevel(logging.INFO)
        elif level == 'W':
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)

        # Set the filename to save the logs to
        self.logger_base_dir = os.path.join(settings.BASE_DIR, 'logs', 'data_science')
        handler = logging.FileHandler(os.path.join(self.logger_base_dir, f"{logger_name}.log"))
        
        # Set the formatter
        handler.setFormatter(self.set_format())

        # Apply the handler created to this logger
        self.logger.addHandler(handler)


    def set_format(self) -> logging.Formatter:
        """Sets the format that the logger should use when making an entry."""
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        return formatter

    
    @staticmethod
    def clear_log(filepath: str) -> None:
        """Clears the log file at filepath."""
        with open(filepath, 'w'):
            pass