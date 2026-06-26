import logging
import os
import inspect
import sys

from tabulate import tabulate
from itertools import zip_longest

class My_logger:
    def __init__(self,
        logger_name = 'experiment',
    ):
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        self.logger = self.get_log(log_dir=log_dir)
        self.head_fmt_setting, self.body_fmt_setting, self.tail_fmt_setting, self.key_fmt_setting, self.blank_fmt_setting = self.get_format()
        self.file_handler, self.console_handler = self.get_add_handler(file_log_parth=log_dir, logger_name=logger_name)

    def get_log(self, log_dir):
        # Use the caller's parent directory as the logger name.
        frame = inspect.stack()[2]
        caller_file = frame.filename
        parent_dir = os.path.basename(os.path.dirname(caller_file))
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger(parent_dir)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        return logger


    def get_format(self):
        head_fmt = (
            "\r\n" + "=" * 60 +
            "-- %(asctime)s | Logger: %(name)s | %(filename)s:%(lineno)d | %(levelname)s -- \r\n%(message)s"
        )
        body_fmt = (
            "-- %(asctime)s | %(name)s | %(filename)s:%(lineno)d | %(levelname)s -- \r\n%(message)s"
        )
        tail_fmt = (
            "\r\n-- %(asctime)s | %(name)s | %(filename)s:%(lineno)d | %(levelname)s \r\n%(message)s -- \r\n" + "=" * 60
        )
        key_fmt = (
            "*" * 60 +
            " \r\n-- %(asctime)s | %(name)s | %(filename)s:%(lineno)d | %(levelname)s -- \r\n%(message)s \r\n" + "*" * 60
        )
        blank_fmt = (
                # "%(message)s\t\t -- %(filename)s:%(lineno)d"
                "%(message)s\t\t"
        )

        head_formater = My_Formatter(fmt=head_fmt, datefmt='%m/%d/%Y %I:%M:%S')
        body_formater = My_Formatter(fmt=body_fmt, datefmt='%m/%d/%Y %I:%M:%S')
        tail_formater = My_Formatter(fmt=tail_fmt, datefmt='%m/%d/%Y %I:%M:%S')
        key_formater = My_Formatter(fmt=key_fmt, datefmt='%m/%d/%Y %I:%M:%S')
        blank_formater = My_Formatter(fmt=blank_fmt, datefmt='%m/%d/%Y %I:%M:%S')

        head = dict(fmt=head_fmt, formater=head_formater)
        body = dict(fmt=body_fmt, formater=body_formater)
        tail = dict(fmt=tail_fmt, formater=tail_formater)
        key = dict(fmt=key_fmt, formater=key_formater)
        blank = dict(fmt=blank_fmt, formater=blank_formater)

        return head, body, tail, key, blank


    def get_add_handler(self, file_log_parth='./logs', logger_name="None_logger"):
        file_handler = logging.FileHandler(os.path.join(file_log_parth, logger_name + '.log'))
        console_handler = logging.StreamHandler(sys.stdout)

        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        return file_handler, console_handler

    def use_format(self, pos):
        assert pos in ['head', 'body', 'tail', 'key', 'blank'], f"Need pos in 'head', 'body', 'tail', 'key' or 'blank', but get {pos}"

        if pos == 'head':
            self.file_handler.setFormatter(logging.Formatter(self.head_fmt_setting['fmt']))
            self.console_handler.setFormatter(self.head_fmt_setting['formater'])
        elif pos == 'body':
            self.file_handler.setFormatter(logging.Formatter(self.body_fmt_setting['fmt']))
            self.console_handler.setFormatter(self.body_fmt_setting['formater'])
        elif pos == 'tail':
            self.file_handler.setFormatter(logging.Formatter(self.tail_fmt_setting['fmt']))
            self.console_handler.setFormatter(self.tail_fmt_setting['formater'])
        elif pos == 'key':
            self.file_handler.setFormatter(logging.Formatter(self.key_fmt_setting['fmt']))
            self.console_handler.setFormatter(self.key_fmt_setting['formater'])
        elif pos == 'blank':
            self.file_handler.setFormatter(logging.Formatter(self.blank_fmt_setting['fmt']))
            self.console_handler.setFormatter(self.blank_fmt_setting['formater'])

    def debug(self, msg, pos='body'):
        self.format_log(logging.DEBUG, msg, pos=pos)

    def info(self, msg, pos='body'):
        self.format_log(logging.INFO, msg, pos=pos)

    def error(self, msg, pos='body'):
        self.format_log(logging.DEBUG, msg, pos=pos)

    def format_log(self, level, msg, pos):
        self.use_format(pos)
        self.logger.log(level, msg, stacklevel=3)


    @staticmethod
    def format_message_multicolumn(config_dict, content='param'):
        assert isinstance(config_dict, dict), f"Config should be a dict, but got {type(config_dict)}"
        assert content in ['param', 'results']

        num_columns = 3 if content == 'param' else 1

        items = list(config_dict.items())
        grouped = list(zip_longest(*[items[i::num_columns] for i in range(num_columns)], fillvalue=("", "")))

        table_data = []
        for row in grouped:
            flat_row = []
            for key, value in row:
                flat_row.extend([key, value])
            table_data.append(flat_row)

        headers = []
        for _ in range(num_columns):
            if content == 'param':
                headers.extend(["Parameter", "Value"])
            elif content == 'results':
                headers.extend(["Dataset", "Results"])

        return tabulate(table_data, headers=headers, tablefmt="fancy_grid")


class My_Formatter(logging.Formatter):
    COLOR_MAP = {
        logging.INFO: "\033[92m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"
