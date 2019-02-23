import logging

def add_logging_level(level_value, level_name):
    """
    Add a new level to the logging module.  Allows that level to be called
    via the lowercased method name, e.g., level INFO_CV is logged by info_cv.

    Args:
        level_value (int): the value of the level.
        level_name (str): the name for the level.

    Returns:
        None

    """
    def log_at_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_value):
            self._log(level_value, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_value, message, *args, **kwargs)

    logging.addLevelName(level_value, level_name)
    setattr(logging, level_name, level_value)
    setattr(logging.getLoggerClass(), level_name.lower(), log_at_level)
    setattr(logging, level_name.lower(), log_to_root)


add_logging_level(logging.INFO - 1, "PREDICTOR") # used for Predictor logging
add_logging_level(logging.INFO - 2, "CV") # used for supervised CV object logging
add_logging_level(logging.INFO - 3, "SUPERVISED") # used for single supervised model logging

logger = logging.getLogger("representation_learning_for_transcriptomics")
