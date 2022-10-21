from datetime import datetime
import inspect
from flask import session

from apps.proc import constants


class Logger(object):
    def __init__(self, filename):
        self.filename = filename

    def _write_log(self, level, msg):
        flat_msg = '<'+msg[0]+'> '

        if len(msg) > 1:
            for elem in msg[1:]:
                if type(elem) == str:
                    flat_msg += elem + ' '
                else:
                    flat_msg += str(elem) + ' '
        mess = "{0} [{1}] {2}".format(datetime.now().strftime('%m-%d-%y %H:%M:%S'), level, flat_msg)
        if level == 'ERROR':
            print(constants.bcolors.FAIL + mess + constants.bcolors.ENDC)
        elif level == 'WARN':
            print(constants.bcolors.WARNING + mess + constants.bcolors.ENDC)
        elif level == 'INFO':
            print(constants.bcolors.OKGREEN + mess + constants.bcolors.ENDC)
        elif level.startswith('DEBUG'):
            print(constants.bcolors.OKBLUE + mess + constants.bcolors.ENDC)
        else:
            print(mess)
        with open(self.filename, 'a', encoding='utf-8') as log_file:
            log_file.write(mess + '\n')


    def critical(self, *msg):
        self._write_log('CRITICAL', msg)

    def error(self, *msg):
        self._write_log("ERROR", msg)

    def warn(self, *msg):
        self._write_log("WARN", msg)

    def info(self, *msg):
        self._write_log("INFO", msg)

    def debug(self, *msg):
        self._write_log("DEBUG "+str(inspect.stack()[1].function), msg)

