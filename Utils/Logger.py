import os
from enum import Enum

from logging import Logger, FileHandler, StreamHandler, Formatter
from datetime import datetime
from colorama import Fore, Style


class ColorScheme(str, Enum):
    white = Fore.WHITE
    red = Fore.RED
    blue = Fore.BLUE
    green = Fore.GREEN


class CustomLogger(Logger):
    def __init__(self, logging_level=None, log_name="untitled", write_to_file=False, logs_path="./Logs", style=None):
        super().__init__(name=log_name)

        self.level = "ERROR" if logging_level is None else logging_level.upper()
        self.setLevel(self.level)

        self.formatter = None
        self.set_formatter(style, updateHandlers=False)

        self.toFile = write_to_file
        self.path = None
        self.set_new_file_attrs(logName=log_name, logsPath=logs_path, updateHandlers=False)

        self.update_stream_handler(init=True)
        self.update_file_handler(init=True)


    @staticmethod
    def colorize(msg, color):
        return ColorScheme[color].value + msg + Style.RESET_ALL


    @staticmethod
    def getID():
        return ("%s" % datetime.now()).replace(" ", "_").replace(":", "-")


    def debug(self, msg, *args, **kwargs):
        color = kwargs.pop("color", "blue")
        msg = self.colorize(msg, color)

        super().debug(msg, *args, **kwargs)


    def info(self, msg, *args, **kwargs):
        color = kwargs.pop("color", "white")
        msg = self.colorize(msg, color)

        super().info(msg, *args, **kwargs)


    def error(self, msg, *args, **kwargs):
        color = kwargs.pop("color", "red")
        msg = self.colorize(msg, color)

        super().error(msg, *args, **kwargs)


    def warning(self, msg, *args, **kwargs):
        color = kwargs.pop("color", "red")
        msg = self.colorize(msg, color)

        super().warning(msg, *args, **kwargs)


    def critical(self, msg, *args, **kwargs):
        color = kwargs.pop("color", "red")
        msg = self.colorize(msg, color)

        super().critical(msg, *args, **kwargs)


    def set_formatter(self, style=None, updateHandlers=True):
        if style is None:
            self.formatter = Formatter("%(message)s")
        else:
            self.formatter = Formatter(style)

        if updateHandlers:
            self.update_stream_handler()
            self.update_file_handler()


    def set_new_file_attrs(self, logName="untitled", logsPath="./Logs", includeTimeID=True, updateHandlers=True):
        os.makedirs(logsPath, exist_ok=True)

        self.timeID = "_{}".format(self.getID()) if includeTimeID else ""

        logFileName = "{}{}.log".format(logName, self.timeID)
        self.path = os.path.join(logsPath, logFileName)

        if updateHandlers:
            self.update_file_handler()


    def update_stream_handler(self, init=False):
        streamHandler = StreamHandler()
        streamHandler.setFormatter(self.formatter)
        streamHandler.setLevel(self.level)
        if init:
            self.addHandler(streamHandler)
        else:
            self.handlers[0] = streamHandler


    def update_file_handler(self, init=False):
        if self.toFile:
            fileHandler = FileHandler(self.path, "a", encoding="utf-8")
            fileHandler.setFormatter(self.formatter)
            fileHandler.setLevel(self.level)
            if init:
                self.addHandler(fileHandler)
            else:
                self.handlers[-1] = fileHandler


    def update_file_attrs(self, logName, logPath="./Logs"):
        self.set_new_file_attrs(logName, logPath)
        self.update_file_handler()


def test():
    loggerInfo = CustomLogger(log_name="info", write_to_file=True, logging_level="info")
    loggerDebug = CustomLogger(log_name="debug", write_to_file=True, logging_level="debug")

    loggerInfo.info("Info logger has been created")
    loggerDebug.debug("Debug logger has been created")

    loggerInfo.debug("Level debug is not working")

    loggerInfo.set_formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)s\n%(message)s")

    loggerInfo.info("New style has been applied", color="green")

    loggerInfo.set_new_file_attrs(logName="noabra", updateHandlers=False)
    loggerInfo.error("Houston, we have a %s", CustomLogger.colorize("bit of a problem", "blue"))

    loggerInfo.update_file_handler()
    loggerInfo.info("Writing to the new file")


if __name__ == "__main__":
    test()
