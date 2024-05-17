import logging
import re
from typing import Optional

import rich.logging
import rich.text

MAIN_LOGGER_NAME = "TRAINER"

REGEX_PATTERN = r"((?:\w+\.)*\w+)=([-]?\d+(?:\.\d+)?(?:e[-+]?\d+)?|\w+|\([^)]*\))"


def fetch_main_logger(apply_basic_config=False):
    logger = logging.getLogger(MAIN_LOGGER_NAME)
    if apply_basic_config:
        configure_logger(logger)
    return logger


def configure_logger(logger, custom_format=None, level=logging.INFO, propagate=False, show_path=False):
    logger.propagate = propagate

    for handler in logger.handlers:
        logger.removeHandler(handler)

    format = f"%(module)s:%(funcName)s:%(lineno)d | %(message)s" if custom_format is None else custom_format
    log_formatter = logging.Formatter(format)

    rich_handler = rich.logging.RichHandler(
        markup=True,
        rich_tracebacks=True,
        omit_repeated_times=True,
        show_path=False,
        log_time_format=lambda dt: rich.text.Text.from_markup(f"[red]{dt.strftime('%y-%m-%d %H:%M:%S.%f')[:-4]}"),
    )
    rich_handler.setFormatter(log_formatter)
    logger.addHandler(rich_handler)

    logger.setLevel(level)


def drill_to_key_and_set(_dict, key, value) -> None:
    # Need to split the key by "." and traverse the config to set the new value
    split_key = key.split(".")
    entry_in_config = _dict
    for subkey in split_key[:-1]:
        entry_in_config = entry_in_config[subkey]
    entry_in_config[split_key[-1]] = value


def update_config_with_cli_args_(config, variables):
    for key, value in variables.items():
        try:
            value = eval(value)
        except NameError:
            pass
        drill_to_key_and_set(config, key=key, value=value)


def default_get_config(config_key: str, pop_key: str, preset_configs: dict, cli_args: Optional[str] = "") -> dict:
    # Extract the key-value pairs from the config string which has the format
    # "key1=value1 key2=value2 ..."
    cli_args = cli_args.strip() if cli_args is not None else ""
    matches = re.findall(REGEX_PATTERN, cli_args)
    # Create a dictionary to store the extracted values
    variables = {key: value for key, value in matches}
    config = preset_configs[variables.pop(pop_key)]()
    update_config_with_cli_args_(config, variables)
    return {config_key: config}
