from collections import deque
from glob import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

BOOLEAN_TRUE = ["true", "t", "1", "yes", "y"]
BOOLEAN_FALSE = ["false", "f", "0", "no", "n"]
BOOLEAN_VALUES = BOOLEAN_TRUE + BOOLEAN_FALSE
MESSAGE_STATUS = {"SUCCESS": 0, "FAIL": 1}

class bcolors:
    BOLD = '\033[1m'
    DISABLE = '\033[02m'
    ITALIC = '\033[03m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[05m'
    HIGHLIGHT = INVERT = '\033[07m'
    INVISIBLE = '\033[08m'
    STRIKETHROUGH = '\033[09m'
    DOUBLE_UNDERLINE = '\033[21m'
    OVERLINE = '\033[53m'
    PALETTE_0 = DARK_GREY = '\033[30m'
    PALETTE_1 = DARK_RED = '\033[31m'
    PALETTE_2 = DARK_GREEN = '\033[32m'
    PALETTE_3 = DARK_YELLOW = '\033[33m'
    PALETTE_4 = DARK_BLUE = '\033[34m'
    PALETTE_5 = DARK_PINK = '\033[35m'
    PALETTE_6 = DARK_AQUA = '\033[36m'
    PALETTE_7 = DARK_WHITE = '\033[37m'
    PALETTE_8 = GREY = '\033[90m'
    PALETTE_9 = RED = '\033[91m'
    PALETTE_10 = GREEN = '\033[92m'
    PALETTE_11 = YELLOW = '\033[93m'
    PALETTE_12 = BLUE = '\033[94m'
    PALETTE_13 = PINK = '\033[95m'
    PALETTE_14 = AQUA = '\033[96m'
    PALETTE_15 = WHITE = '\033[97m'
    BG_PALETTE_0 = BG_DARK_GREY = '\033[40m'
    BG_PALETTE_1 = BG_DARK_RED = '\033[41m'
    BG_PALETTE_2 = BG_DARK_GREEN = '\033[42m'
    BG_PALETTE_3 = BG_DARK_YELLOW = '\033[43m'
    BG_PALETTE_4 = BG_DARK_BLUE = '\033[44m'
    BG_PALETTE_5 = BG_DARK_PINK = '\033[45m'
    BG_PALETTE_6 = BG_DARK_AQUA = '\033[46m'
    BG_PALETTE_7 = BG_DARK_WHITE = '\033[47m'
    BG_PALETTE_8 = BG_GREY = '\033[100m'
    BG_PALETTE_9 = BG_RED = '\033[101m'
    BG_PALETTE_10 = BG_GREEN = '\033[102m'
    BG_PALETTE_11 = BG_YELLOW = '\033[103m'
    BG_PALETTE_12 = BG_BLUE = '\033[104m'
    BG_PALETTE_13 = BG_PINK = '\033[105m'
    BG_PALETTE_14 = BG_AQUA = '\033[106m'
    BG_PALETTE_15 = BG_WHITE = '\033[107m'
    BG_BLACK = '\033[48m'  # default background color
    ENDC = '\033[0m'


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print_msg(f"Function '{func.__name__}' took {end_time - start_time:.6f} seconds to execute.", "BLUE")
        return result
    return wrapper


def make_boolean(v: str, empty_str_as_false: Optional[bool] = False) -> bool:
    """
    make a boolean value from a string
    supported values:
        values in BOOLEAN_VALUES
    if empty_str_as_false=True an empty string is interpreted as False

    Parameters:
    -----------
    v
        string that should be converted to a boolean
    empty_str_as_false
        if empty_str_as_false=True an empty string is interpreted as False

    Returns:
    --------
    bool
        boolean value from v
    """
    # treat empty string as "False"
    if empty_str_as_false and len(v) == 0:
        return False

    lower = str(v).lower()

    if lower in BOOLEAN_TRUE:
        return True
    elif lower in BOOLEAN_FALSE:
        return False
    else:
        raise ValueError(f"unsupported string for converting to boolean: '{lower}'")


def color_str(msg: Any, color: Union[str, List[str]]) -> str:
    """
    adds the color options inside 'color' to the 'msg' parameter

    Parameters
    ----------
    msg: any
        requirement is that 'msg' can be converted by str()
    color: list or tuple of str
        strings of variable names in 'bcolors' class
    """
    coda = "" if repr(msg).endswith(bcolors.ENDC[1:]) else bcolors.ENDC

    if isinstance(color, str):
        return getattr(bcolors, color) + str(msg) + coda
    return "".join([getattr(bcolors, c) for c in color]) + str(msg) + coda


def print_msg(msg: Any, color: Union[str, List[str]] = "ENDC", end: str = "\n") -> None:
    """
    print a message in a colorful way

    Parameters:
    -----------
    msg : str | object with '__print__' or '__repr__' function
        message to print
    color : str, list of str | default=ENDC
        color that needs to be defined within bcolors
    end: str | default=new line
        string append after the last value

    Returns:
    --------
    no return
    """
    assert isinstance(color, (str, list)), "'color' has to be a str or a list of str!"
    assert isinstance(end, str), "'end' has to be a str!"

    cstr = color_str(msg, color)
    print(cstr, end=end)


def return_value(status: int,
                 property: str = "",
                 value: Any = None,
                 message: str = "",
                 version: str = "") -> Dict[str, Any]:
    """
    A function that generates a dictionary with at minimum 'status' as key with value from MESSAGE_STATUS,
    optionally you can add a message, or a property string and a property value, or the version of the
    running script.

    Parameters
    ----------
    status:
        a key in MESSAGE status that represents an error code, or success
    property:
        the identity of a property as a string
    value:
        the value of the property referenced in 'property'
    message:
        an error or log message
    version:
        for larger scripts that have version numbers, the version of the running script.

    Returns
    -------
    val:
        a dictionary with at minimum 'status' as a key
    """
    if status not in MESSAGE_STATUS.values():
        print_msg(f"Warning, status: '{status}' should be a value in MESSAGE_STATUS", "YELLOW")
    # create return value
    val = {"status": status}
    if property:
        val["property"] = property  # type: ignore
    if value is not None:
        val["value"] = value
    if message:
        val["message"] = message  # type: ignore
    if version:
        val["version"] = version  # type: ignore

    return val


def get_files(folderpath: str = "",
              filename_pattern: str = "*.png") -> Tuple[List[str], Dict[str, Any]]:
    """
    returns a list of filepaths for a given 'filename_pattern' consumable by 'glob' in 'folderpath' 

    Parameters
    ----------
    folderpath:
        a path to a folder containing a camera.json and associated images
    filename_pattern:
        a filename pattern to use: e.g. "frame*.jpg"
    """
    files = sorted(glob(os.path.join(folderpath, filename_pattern)))
    if not files:
        msg = f"Error!  could not find any image files on the form: '{filename_pattern}' in path: {folderpath}"
        print_msg(msg, "RED")
        return [], return_value(MESSAGE_STATUS['FAIL'], "folderpath", folderpath, msg)

    return files, return_value(MESSAGE_STATUS['SUCCESS'])


def binary_search_order(lst):
    """
    Given a list, reorders the list so that it is in an order similar to the order a binary search goes through the elements
    """
    if not lst:
        return lst

    result = []
    queue = deque([(0, len(lst) - 1)])

    while queue:
        start, end = queue.popleft()
        mid = (start + end) // 2
        result.append(lst[mid])

        if start <= mid - 1:
            queue.append((start, mid - 1))

        if mid + 1 <= end:
            queue.append((mid + 1, end))

    return result
