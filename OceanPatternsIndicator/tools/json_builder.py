import json
from datetime import datetime


class LogError:
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def set_message(self, message):
        self.message = message


class ExecLog:
    def __init__(self):
        self.messages = list()

    def add_message(self, message, run_time=None, formating=False):
        if run_time is not None:
            message += ",  time needed: " + " %s seconds " % run_time
        if formating:
            message = get_log_message(message)
        self.messages.append(message)

    def print_messages(self):
        """
        This function prints on stdout all messages in exec_log class with the index of message
        """
        for i, msg in enumerate(self.messages):
            print(str(i) + " - " + msg)


def get_log_error(err_code, err_message):
    return LogError(err_code, err_message)


def get_exec_log():
    return ExecLog()


def get_log_message(msg):
    """
    This function prints to stdout the message passed as argument with a timestamp as prefix
    @param msg: Message to be printed
    """
    dateTimeObj = datetime.now()
    return "[" + dateTimeObj.strftime("%m/%d/%Y, %H:%M:%S") + "]   -   " + msg


def write_json(**dicts):
    """
    This function writes in a json file the two classes exec_log and log_error
    @param dicts: a list of dicts to insert in a json file
    """
    json_log = dict()
    for jsonDict in dicts:
        json_log[jsonDict] = dicts[jsonDict]
    json.dump(json_log, open("./output.json", "w"), indent=4)