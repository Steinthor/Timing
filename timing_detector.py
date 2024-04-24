#!/usr/bin/env python3

import argparse
from collections import deque
import cv2
import csv
from dataclasses import dataclass, astuple
from glob import glob
import math
import numpy as np
import os
from typing import List, Any, Tuple, Dict, Optional

from tools import get_files, make_boolean, measure_time, print_msg, return_value, MESSAGE_STATUS

MONITOR_MAX_HZ = 61
CAPTURE_FOLDERS = ["capture", "capture_faros", "capture_mocap", "session_", "raw_session_"] # needs to be sanitized
IM_SIZE = 1000
CELL_BOX_LOWER = 250
CELL_BOX_HIGHER = IM_SIZE - CELL_BOX_LOWER
CELL_ROWS = 5
CELL_COLUMNS = 6
CELL_COL_HZ = 3
CELL_COL_SEC = 3
FRAME_THRESHOLD = 50

CORRECTION_SAME_VALUE = 1
CORRECTION_WRONG_VALUE = 2

# COL_VALUES is the interpretation of the binary representation of the values in the cells of a column.
#   The binary permutation of the five cells of a column represent a binary number from 0 to 31 which are
#   the indices of 'COL_VALUES'
# Only certain indices are valid numbers (from 0 to 9), the invalid permutations are given -1 value.
COL_VALUES = [
    0, 1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, 4, 9, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1,
    7, -1, 6, 5
]


@dataclass
class Point:
    x: float
    y: float

    def __array__(self):
        return np.array(astuple(self))

    def __len__(self) -> int:
        return astuple(self).__len__()

    def __getitem__(self, item) -> Any:
        return astuple(self).__getitem__(item)


@dataclass
class Cell:
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point

    def __array__(self, dtype) -> np.ndarray:
        return np.array(astuple(self), dtype=dtype)

    def __len__(self) -> int:
        return astuple(self).__len__()

    def __getitem__(self, item) -> Tuple[Any]:
        return astuple(self).__getitem__(item)


@dataclass
class Column:
    value: int = 0
    grey: int = 0

    def add_cell(self, index: int) -> None:
        self.value |= 2**index

    def add_grey(self, index: int) -> None:
        self.grey |= 2**index

    def get_value(self, grey: bool = False) -> int:
        return COL_VALUES[self.grey | self.value] if grey else COL_VALUES[self.value]

    def __repr__(self) -> str:
        return f"grey: {self.get_value(False)}, on: {self.get_value(True)}"

    def __str__(self) -> str:
        return f"grey: {self.get_value(False)}, on: {self.get_value(True)}"


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


def convert_sec_hz_to_float(value: float = 0,
                            sec: int = 0,
                            hz: int = 0,
                            max_hz: int = MONITOR_MAX_HZ) -> float:
    """
    given 'sec' and 'hz' returns a float where integer portion represents seconds,
        and decimal portion represents a ratio of max_hz
    given 'value' where integer portion is seconds, and decimal portion is a ratio of max_hz,
        returns a float where integer value represents seconds,
        and the decimal value represents a monitor frequency value in base 10,
          example 1: seconds 42, frame 10, max_hz below 100: 42.1
          example 2: seconds 42, frame 10, max_hz above 100, below 1000: 42.01
    """
    order_of_magnitude = math.ceil(math.log(max_hz, 10))
    if value:
        return int(value) + round(
            (value - int(value)) * max_hz / (10**order_of_magnitude), order_of_magnitude)
    return sec + hz / max_hz


def get_pixel_value_interval(img: np.ndarray, cell: Cell, crop: int = 0) -> Tuple[Any, float, float]:
    """
    returns the min and max values of a square defined by the four points in 'corner' in 'img'

    Parameters
    ----------
    img:
        a numpy matrix representing an image
    corner:
        a length 4 list of length 2 lists of floats.
        Assumptions: 1) the square is parallel to the image axes,
                     2) the corners are in CW order with top-left in first index
    crop:
        the number of pixels to remove from each side
    """
    # img[y, x]
    cropped_img = img[cell.top_left.y + crop:cell.bottom_left.y - crop,
                      cell.top_left.x + crop:cell.top_right.x - crop]
    # plt.imshow(cropped_img, cmap='Greys')
    # plt.colorbar()
    # plt.show()
    min = np.amin(cropped_img)
    max = np.amax(cropped_img)
    av = np.average(cropped_img)
    return (av, min, max)


def process_columns(columns: List[Column], base: int = 10) -> Tuple[int, int]:
    time_s = 0
    time_hz = 0

    for i in range(CELL_COL_HZ):
        get_grey = True if columns[i].get_value(grey=True) > 0 else False
        time_hz += base**i * columns[i].get_value(get_grey)

    for i in range(CELL_COL_HZ, len(columns)):
        get_grey = True if columns[i].get_value(grey=True) > 0 else False
        power = i - CELL_COL_HZ
        time_s += base**power * columns[i].get_value(get_grey)
    return time_s, time_hz


def get_time(file,
             H,
             size: Tuple[Any, Any],
             debug: bool = False,
             verbose: bool = False) -> Tuple[int, int, Any, Dict[str, Any]]:
    """
    Calculates the timestamp represented in 'file'

    Parameters:
    -----------
    file:
        a file path to an image with a detected marker in it
    H:
        a perspective transformation matrix
    size:
        the timer image size
    debug:
        whether to display image with markers
    verbose:
        whether to print out additional information to console

    Returns
    -------
    sec_value:
        time in seconds
    hz_value:
        time in Hz
    user input:
        key pressed in opencv window, if debug is enabled, else None.
    """
    if verbose:
        print_msg(file, "GREY")
    img_raw = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if debug:
        cv2.imshow("src", img_raw)
    img_timer = cv2.warpPerspective(img_raw, H, size)
    img = cv2.cvtColor(img_timer, cv2.COLOR_GRAY2RGB)

    # the timer template image is 1k x 1k pixels for convenience
    box_ids = []
    box_lo = CELL_BOX_LOWER
    box_hi = CELL_BOX_HIGHER
    box_cell = Cell(top_left=Point(x=box_lo, y=box_lo),
                    top_right=Point(x=box_hi, y=box_lo),
                    bottom_left=Point(x=box_hi, y=box_hi),
                    bottom_right=Point(x=box_lo, y=box_hi))
    box_marker = [np.array([box_cell], dtype=np.float32)]
    box_ids.append([0])
    white_cell = Cell(top_left=Point(x=300, y=50),
                      top_right=Point(x=725, y=50),
                      bottom_left=Point(x=725, y=225),
                      bottom_right=Point(x=300, y=225))
    white_marker = [np.array([white_cell], dtype=np.float32)]
    box_ids.append([1])
    black_cell = Cell(top_left=Point(x=275, y=280),
                      top_right=Point(x=725, y=280),
                      bottom_left=Point(x=725, y=320),
                      bottom_right=Point(x=275, y=320))
    black_marker = [np.array([black_cell], dtype=np.float32)]
    box_ids.append([2])
    # order of cells: lowest cell of fastest column -> highest cell of fastest col. -> highest cell of slowest column
    cell_size = int((IM_SIZE - 2 * CELL_BOX_LOWER) / CELL_COLUMNS)
    cells = []
    cell_ids = []
    index = len(cell_ids)
    for c in range(0, CELL_COLUMNS):
        right_x = box_hi - c * cell_size
        left_x = box_hi - (c + 1) * cell_size
        for r in range(0, CELL_ROWS):
            bot_y = box_hi - r * cell_size
            top_y = box_hi - (r + 1) * cell_size
            cell = Cell(top_left=Point(x=left_x, y=top_y),
                        top_right=Point(x=right_x, y=top_y),
                        bottom_left=Point(x=right_x, y=bot_y),
                        bottom_right=Point(x=left_x, y=bot_y))
            cells.append(cell)
            cell_ids.append([index])
            index += 1
    cell_markers = [np.array([cell], dtype=np.float32) for cell in cells]

    off_av, off_min, off_max = get_pixel_value_interval(img=img, cell=black_cell, crop=5)
    on_av, on_min, on_max = get_pixel_value_interval(img=img, cell=white_cell)
    cell_off = off_max
    cell_on = on_min
    if debug:
        print_msg(f"black cell: off min: {off_min}, off max: {off_max}, off av: {off_av}", "GREY")
        print_msg(f"white cell: on min : {on_min},  on max : {on_max},  on av : {on_av}", "GREY")
        print_msg(f"cell off: {cell_off}, cell on: {cell_on}", "GREY")

    # loop through the cells to get better cell_on, cell_off values
    for i, cell in enumerate(cells):
        c_av, c_min, c_max = get_pixel_value_interval(img=img, cell=cell, crop=10)
        if c_max >= cell_on:
            if c_min < cell_on and c_min > cell_off:
                cell_on = c_min
        elif c_min <= cell_off:
            if c_max > cell_off and c_max < cell_on:
                cell_off = c_max
        else:
            if debug:
                print_msg(f"did not catch cell: {i} with values: c_min: {c_min}, c_max: {c_max}", ["GREY"])
    if debug:
        print_msg(f"cell off: {cell_off}, cell on: {cell_on}", "PINK")

    # loop through the cells to get the columns
    columns = []
    for i, cell in enumerate(cells):
        col = int(np.floor(i / CELL_ROWS))
        if len(columns) <= col:
            columns.append(Column())

        c_av, c_min, c_max = get_pixel_value_interval(img=img, cell=cell, crop=10)
        if c_max >= cell_on:
            columns[col].add_cell(i % CELL_ROWS)
        elif c_min > cell_off:
            columns[col].add_grey(i % CELL_ROWS)
        else:
            if debug:
                print_msg(f"did not catch cell: {i} with values: c_min: {c_min}, c_max: {c_max}",
                          ["BOLD", "GREY"])

    sec_value, hz_value = process_columns(columns)
    if verbose:
        print_msg(f"time: {sec_value}:{hz_value}", "AQUA")

    markers = box_marker + white_marker + black_marker
    user_input = None
    if debug:
        cv2.aruco.drawDetectedMarkers(img, markers, np.array(box_ids), borderColor=(155, 50, 155))
        cv2.aruco.drawDetectedMarkers(img, cell_markers, np.array(cell_ids), borderColor=(155, 50, 155))
        cv2.imshow("test", img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_thresh = np.clip(gray, cell_off, cell_on)
        cv2.normalize(img_thresh, img_thresh, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("normalized", img_thresh)
        user_input = cv2.waitKey()

    return sec_value, hz_value, user_input, return_value(MESSAGE_STATUS['SUCCESS'])  # ord('\x1b')


def get_corrected_time_values(before_sec: Optional[int] = None,
                              before_hz: Optional[int] = None,
                              after_sec: Optional[int] = None,
                              after_hz: Optional[int] = None,
                              num: int = 0,
                              before_inclusive: bool = False,
                              after_inclusive: bool = False,
                              max_hz: int = MONITOR_MAX_HZ,
                              min_hz: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Helper function, returns a list of 'sec:hz' times of length 'num' between 'before_sec:before_hz' and
    'after_sec:after_hz'
    """
    if (before_sec is None or before_hz is None) and (after_sec is None or after_hz is None):
        msg = "Error!  both before and after times are None"
        print_msg(msg, "RED")
        return np.empty((0,)), return_value(MESSAGE_STATUS['FAIL'], "before_hz", before_hz, msg)
    if num < 0:
        msg = f"Error!  num is a negative value: {num}"
        print_msg(msg, "RED")
        return np.empty((0,)), return_value(MESSAGE_STATUS['FAIL'], "num", num, msg)

    # handle before is None case
    if before_sec is None or before_hz is None:
        after = after_hz / max_hz
        if after > 1.0:
            msg = f"Error!  after_hz: {after_hz} larger than max_hz: {max_hz}"
            print_msg(msg, "RED")
            return np.empty((0,)), return_value(MESSAGE_STATUS['FAIL'], "after_hz", after_hz, msg)
        after += after_sec
        interval = [
            after - (x * min_hz) / max_hz
            for x in range(0 if before_inclusive else 1, num + 2 if after_inclusive else num + 1)
        ]
        interval.reverse()
    # handle after is None case
    elif after_sec is None or after_hz is None:
        before = before_hz / max_hz
        if before > 1.0:
            msg = f"Error!  before_hz: {before_hz} larger than max_hz: {max_hz}"
            print_msg(msg, "RED")
            return np.empty((0,)), return_value(MESSAGE_STATUS['FAIL'], "before_hz", before_hz, msg)
        before += before_sec

        interval = [
            before + (x * min_hz) / max_hz
            for x in range(0 if before_inclusive else 1, num + 2 if after_inclusive else num + 1)
        ]
    # handle both exist case:
    else:
        before = before_hz / max_hz
        if before > 1.0:
            msg = f"Error!  before_hz: {before_hz} larger than max_hz: {max_hz}"
            print_msg(msg, "RED")
            return np.empty((0,)), return_value(MESSAGE_STATUS['FAIL'], "before_hz", before_hz, msg)
        before += before_sec
        after = after_hz / max_hz
        if after > 1.0:
            msg = f"Error!  after_hz: {after_hz} larger than max_hz: {max_hz}"
            print_msg(msg, "RED")
            return np.empty((0,)), return_value(MESSAGE_STATUS['FAIL'], "after_hz", after_hz, msg)
        after += after_sec

        if not before_inclusive:
            num += 1
        if not after_inclusive:
            num += 1
        interval = np.linspace(before, after, num)
        interval = interval[0 if before_inclusive else 1:None if after_inclusive else -1]

    return_list = [(int(x), int(np.rint((x - int(x)) * max_hz))) for x in interval]

    return np.array(return_list), return_value(MESSAGE_STATUS['SUCCESS'])


def fix_time_errors(timings: List[Any],
                    max_hz: int = MONITOR_MAX_HZ,
                    max_relative_value: float = 0.1,
                    verbose: bool = False) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Helper function that goes over the detected timings.
      - Known wrong values are marked with '100' in the 'corrected' column.
      - Values that are outside the interval of its adjacent rows are fixed, and a '1' is set in 'corrected' column.

    Parameters
    ----------
    timings:
        a list of lists: [filepath, time [s], time [hz], corrected]
            filepath: the original file that was used to get the time
            time [s]: the detected time in seconds
            time [hz]: the detected time in Herz
            corrected: whether the time value has been corrected
    max_hz:
        the max number of frames of a monitor per second
    max_relative_value:
        the time period in an interval divided by the number of frames in the interval should be
        below this value

    Returns
    -------
    modified timings
    """

    def check_time_intervals(order: np.ndarray,
                             time_intervals: np.ndarray,
                             initial: bool = False) -> np.ndarray:
        for i, (start, end, error) in enumerate(time_intervals):
            # print(f"start: {start} - {order[start][1]}:{order[start][2]}, ",
            #       f"end: {end} - {order[end][1]}:{order[end][2]}, error: {error}")
            if start == end:
                # print("problem: just one value")
                time_intervals[i][2] = -1
                continue
            if order[start][1] < 0 or order[start][2] < 0 or order[end][1] < 0 or order[end][2] < 0:
                # print("problem: values are negative")
                time_intervals[i][2] = -1
                continue
            series, series_counts = np.unique(order[start:end + 1, 1:3], axis=0, return_counts=True)
            if np.all(series_counts != 1):
                # print("problem: no unique values")
                time_intervals[i][2] = -1
                continue
            start_val = series[0][0] + series[0][1] / max_hz
            end_val = series[len(series) - 1][0] + series[len(series) - 1][1] / max_hz
            av_rel_val = (end_val - start_val) / (end - start)
            if av_rel_val > max_relative_value:
                # print(f"problem: average relative value too high: {av_rel_val}")
                time_intervals[i][2] = -1
                continue
            if initial and i == len(time_intervals) - 1 and end != len(order) - 1:
                # print(f"problem: the largest value is not at the end of the index")
                time_intervals[i][2] = -1
            # print("series should be fine")
            # pprint([(x, y) for x, y in zip(series, series_counts)])
            # print(f"average relative value: {av_rel_val}")

        return time_intervals

    def graph_sweep(order: np.ndarray, order_intervals: np.ndarray) -> Dict[str, Any]:
        """
        Helper function that finds the longest connection of monotonically increasing intervals and
        marks all other intervals as containing errors.
        """
        graph = []
        for i, interval in enumerate(order_intervals):
            error = interval[2]
            if error != 0:
                continue
            order_intervals[i][2] = -2
            start_sec = order[interval[0]][1]
            start_hz = order[interval[0]][2]
            start = convert_sec_hz_to_float(sec=start_sec, hz=start_hz)
            for j, series in enumerate(graph):
                before_sec = order[series[-1][1]][1]
                before_hz = order[series[-1][1]][2]
                before = convert_sec_hz_to_float(sec=before_sec, hz=before_hz)
                if before < start:
                    graph[j].append(interval.tolist())
            graph.append([interval.tolist()])

        max_sum = -1
        max_i = -1
        for i, series in enumerate(graph):
            series_sum = sum(interval[1] - interval[0] + 1 for interval in series)
            if series_sum > max_sum:
                max_sum = series_sum
                max_i = i
        if max_i == -1:
            return return_value(MESSAGE_STATUS['FAIL'],
                                message="graph_sweep: only error intervals were found")
        j = 0
        for interval in graph[max_i]:
            while j < len(order_intervals) and not (interval == order_intervals[j]).all():
                j += 1
            order_intervals[j][2] = 0
            j += 1

        # for interval in order_intervals:
        #     print(f"{interval}: start: {order[interval[0]][1]}:{order[interval[0]][2]}, ",
        #           f"end: {order[interval[1]][1]}:{order[interval[1]][2]}, ", f"error: {interval[2]}")

        return return_value(MESSAGE_STATUS['SUCCESS'])

    def fix_order_intervals(order: np.ndarray,
                            order_intervals: np.ndarray,
                            verbose: bool = False) -> Dict[str, Any]:
        # print("order_intervals")
        # pprint([(f"{x[0]} - {order[x[0]][1]}:{order[x[0]][2]}", f"{x[1]} - {order[x[1]][1]}:{order[x[1]][2]}",
        #          x[2]) for x in order_intervals])
        reply = graph_sweep(order, order_intervals)
        if reply['status'] != MESSAGE_STATUS['SUCCESS']:
            print_msg(f"Warning!  graph sweep only found order intervals with errors!", "YELLOW")
        i = 0
        num = 0
        before_sec = None
        before_hz = None
        before_index = 0
        after_sec = None
        after_hz = None
        after_index = 0
        process_after = False
        while i < len(order_intervals):
            # print(
            #     f"i: {i}: ({order_intervals[i][0]},{order_intervals[i][1]}),{order_intervals[i][2]}",
            #     f" - ({order[order_intervals[i][0]][1]}:{order[order_intervals[i][0]][2]},",
            #     f"{order[order_intervals[i][1]][1]}:{order[order_intervals[i][1]][2]})",
            #     f", num: {num}",
            # )
            if order_intervals[i][2] != 0:
                process_after = True
                num += order_intervals[i][1] - order_intervals[i][0] + 1
                i += 1
                continue
            if process_after:
                after_sec = order[order_intervals[i][0]][1]
                after_hz = order[order_intervals[i][0]][2]
                after_index = order_intervals[i][0]
                before_inclusive = 0 if before_index == 0 else 1
                # if before_sec and after_sec:
                num = after_index - before_index - before_inclusive
                if before_index == 0:
                    values, reply = get_corrected_time_values(before_sec=None,
                                                              before_hz=None,
                                                              after_sec=after_sec,
                                                              after_hz=after_hz,
                                                              num=num,
                                                              before_inclusive=False,
                                                              after_inclusive=False,
                                                              max_hz=max_hz)
                else:
                    values, reply = get_corrected_time_values(before_sec,
                                                              before_hz,
                                                              after_sec,
                                                              after_hz,
                                                              num=num,
                                                              before_inclusive=False,
                                                              after_inclusive=False,
                                                              max_hz=max_hz)
                if reply['status'] != MESSAGE_STATUS['SUCCESS']:
                    return reply
                # print(
                #     f"num: {num}, before index: {before_index}, after index: {after_index}, values: {len(values)}"
                # )
                values = np.append(values, np.ones((len(values), 1)) * CORRECTION_WRONG_VALUE, axis=1)
                order[before_index + before_inclusive:after_index,
                      1:4] = values[0:after_index - before_index - before_inclusive]
                process_after = False
            before_sec = order[order_intervals[i][1]][1]
            before_hz = order[order_intervals[i][1]][2]
            before_index = order_intervals[i][1]
            num = 0
            i += 1
        if process_after:
            after_index = order_intervals[-1][1] + 1
            # print(f"num: {num}, before index: {before_index}, after index: {after_index}")
            values, reply = get_corrected_time_values(before_sec=before_sec,
                                                      before_hz=before_hz,
                                                      after_sec=None,
                                                      after_hz=None,
                                                      num=num,
                                                      before_inclusive=False,
                                                      after_inclusive=False,
                                                      max_hz=max_hz,
                                                      min_hz=2)
            if reply['status'] != MESSAGE_STATUS['SUCCESS']:
                return reply
            values = np.append(values, np.ones((len(values), 1)) * CORRECTION_WRONG_VALUE, axis=1)
            order[before_index + 1:after_index, 1:4] = values[0:after_index - before_index - 1]

        return return_value(MESSAGE_STATUS['SUCCESS'])

    order = np.array([(i, row[1], row[2], row[3]) for i, row in enumerate(timings)])
    order[:, 2] = np.clip(order[:, 2], 0, max_hz)

    count = 0
    time_length = -1
    while True:
        count += 1
        # sort timings based on time value, first seconds, then hz, then file order
        time_order = order[order[:, 0].argsort()]
        time_order = time_order[time_order[:, 2].argsort(kind='mergesort')]
        time_order = time_order[time_order[:, 1].argsort(kind='mergesort')]
        # get intervals of monotonically increasing order
        time_intervals = []
        start = 0
        for i, (t1, t2) in enumerate(zip(time_order, time_order[1:])):
            if t2[0] - t1[0] == 1:
                continue
            time_intervals.append((time_order[start][0], time_order[i][0], 0))
            start = i + 1
        end = i if i > start else i + 1
        time_intervals.append((time_order[start][0], time_order[end][0], 0))
        time_intervals = np.array(time_intervals)
        if time_length != -1 and len(time_intervals) == time_length:
            msg = "Error!  function 'fix_time_errors' is stuck in a loop!  Manual fix of timing.csv is necessary"
            print_msg(msg, "RED")
            return timings, return_value(MESSAGE_STATUS['FAIL'], "time_intervals", time_intervals)

        time_length = len(time_intervals)
        if time_length == 1:
            break
        time_intervals = check_time_intervals(order, time_intervals, initial=True if count == 1 else False)
        # print("time_intervals")
        # pprint([(f"{x[0]} - {order[x[0]][1]}:{order[x[0]][2]}", f"{x[1]} - {order[x[1]][1]}:{order[x[1]][2]}",
        #          x[2]) for x in time_intervals])
        # return intervals to file order
        order_intervals = time_intervals[time_intervals[:, 0].argsort()]
        reply = fix_order_intervals(order, order_intervals, verbose=verbose)
        if reply['status'] != MESSAGE_STATUS['SUCCESS']:
            return timings, reply
        # print("order_intervals")
        # pprint([(f"{x[0]} - {order[x[0]][1]}:{order[x[0]][2]}", f"{x[1]} - {order[x[1]][1]}:{order[x[1]][2]}",
        #          x[2]) for x in order_intervals])

    # find repeating time values and give them different values
    order_times = order[:, 1:3]
    error_times = order[:, 3]
    time, time_counts = np.unique(order_times, axis=0, return_counts=True)
    for i, _ in enumerate(time):
        if time_counts[i] == 1:
            continue
        time_filter = np.where((order_times[:, 0] == time[i][0]) * (order_times[:, 1] == time[i][1]))
        num = len(time_filter[0])
        # print(f"i: {i}, num: {num}, time: {_}, time_counts: {time_counts[i]}")
        if i == 0:
            current_sec = None
            current_hz = None
            after_sec = time[i][0]
            after_hz = time[i][1]
            num += -1
        elif i == len(time) - 1:
            current_sec = time[i][0]
            current_hz = time[i][1]
            after_sec = None
            after_hz = None
            num += -1
        else:
            current_sec = time[i][0]
            current_hz = time[i][1]
            after_sec = time[i + 1][0]
            after_hz = time[i + 1][1]
        # print(f"current: {current_sec}:{current_hz}, after: {after_sec}:{after_hz}")

        values, reply = get_corrected_time_values(current_sec,
                                                  current_hz,
                                                  after_sec,
                                                  after_hz,
                                                  num=num,
                                                  max_hz=max_hz,
                                                  min_hz=2,
                                                  before_inclusive=True)
        if reply['status'] != MESSAGE_STATUS['SUCCESS']:
            return timings, reply
        error_times[time_filter] = CORRECTION_SAME_VALUE
        error_times[time_filter[0][0]] = 0
        order_times[time_filter] = values

    timings = [(fp[0], times[0], times[1], err) for fp, times, err in zip(timings, order_times, error_times)]

    return timings, return_value(MESSAGE_STATUS['SUCCESS'])


def write_timings_to_csv(folder_path: str, filename: str, timings: List[Any]) -> Dict[str, Any]:
    # save the timings in a csv file.
    with open(os.path.join(folder_path, filename), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("filename", "time [s]", "time [Hz]", "corrected"))
        writer.writerows(timings)
    msg = f"{filename} written to: {folder_path}"
    print_msg(msg, "GREEN")
    return return_value(MESSAGE_STATUS['SUCCESS'], message=msg)


def get_marker_images(inpath: str,
                    aruco_detector: Any,
                    filename_pattern: str = "*.png",
                    verbose: bool = False,
                    debug: bool = False) -> Tuple[Dict[Any, Any], Dict[str, Any]]:
    """
    Goes through subfolders of inpath, detects markers and returns the timings and filepath of the
    folder with the most found markers

    Parameters
    ----------
    inpath:
        folder to search for scans
    aruco_detector:
        aruco detector to use to detect markers
    verbose:
        print additional information to console
    debug:
        generate more data
    """

    marker_images = []
    marker_path = []

    files, reply = get_files(folderpath=inpath, filename_pattern=filename_pattern)
    if not files or reply['status'] != MESSAGE_STATUS['SUCCESS']:
        return marker_images, marker_path, reply

    # loop through the images, find the ones with all four markers detected
    print_msg(f"detecting markers in {inpath}...", "BLUE")
    counter = 0
    markers_before_threshold = False
    marker_id_test = set((0, 1, 2, 3))
    images = {}
    for file in files:
        if counter > FRAME_THRESHOLD and not markers_before_threshold:
            break
        counter += 1
        img_raw = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        thresholds = binary_search_order(range(250, 100, -5))
        for lower_threshold in thresholds[:10]:
            found_markers = False
            _, img_thresh = cv2.threshold(img_raw, lower_threshold, 255, cv2.THRESH_TOZERO)

            (corners, ids, rejected) = aruco_detector.detectMarkers(img_thresh)
            if not corners:
                # if debug:
                # print_msg(f"No markers found in: {file}", "YELLOW")
                # cv2.aruco.drawDetectedMarkers(img_thresh, rejected)
                # cv2.imshow("rejected", img_thresh)
                # reply = cv2.waitKey()
                # if reply == ord('\x1b'):
                #     break
                continue
            if len(corners) > 2:
                markers_before_threshold = True
            if len(corners) < 4:
                continue
            id_set = set(tuple(ids.ravel()))
            if marker_id_test != id_set:
                if debug:
                    print_msg("marker ids are not correct!", "YELLOW")
                continue
            found_markers = True
            break

        if not found_markers:
            if debug:
                print_msg(f"did not find markers in: {file}", "YELLOW")
            continue
        if debug:
            print_msg(f"found markers for file: {file}", "GREEN")
        images[file] = {}
        for corner, id in zip(corners, ids):
            id = id[0]
            images[file][id] = corner
    if not images:
        print_msg(f"no markers found in {inpath}", "YELLOW")

    print_msg(f"found {len(images)} images with markers", "DARK_GREEN" if len(images) > 0 else "YELLOW")

    return images, return_value(MESSAGE_STATUS['SUCCESS'])


@measure_time
def mocap_timing_detector(inpath: str,
                          template: str,
                          filename_pattern: str = "*.png",
                          max_hz: int = MONITOR_MAX_HZ,
                          use_existing_raw: bool = True,
                          debug: bool = False,
                          verbose: bool = False) -> Dict[str, Any]:
    """
    generates a 'timings.csv' file in the subfolder containing images with the mocap timing app.

    Parameters
    ----------
    inpath:
        a mocap session folder, a scancode folder or a specific file path to a camera.json
    template:
        a file path to the template.png image generated by the mocap timing app
    max_hz:
        the maximum frames from the monitor per second
    timpath:
        optional filepath to a 'timings_raw.csv' file to skip processing the images each time
    debug:
        display images for analysis
    verbose:
        print additional information to console
    """
    if not os.path.isfile(template):
        msg = f"Error!  template: {template} not found"
        print_msg(msg, "RED")
        return return_value(MESSAGE_STATUS['FAIL'], "template", template, msg)

    raw_files = sorted(glob(os.path.join(inpath, "**", "timings_raw.csv"), recursive=True))
    if use_existing_raw and raw_files:
        for raw_file in raw_files:
            timings = []
            # open the timings in a csv file.
            with open(raw_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    timings.append([row[0], int(row[1]), int(row[2]), int(row[3])])

            # do some error checking on the timing result using adjacent time result as a reference
            timings, reply = fix_time_errors(timings, max_hz=max_hz, verbose=verbose)
            if reply['status'] != MESSAGE_STATUS['SUCCESS']:
                return reply
            write_timings_to_csv(os.path.dirname(raw_file), "timings.csv", timings)

        return return_value(MESSAGE_STATUS['SUCCESS'], message="existing timings_raw.csv processed")

    aruco_dict_str = cv2.aruco.DICT_APRILTAG_16h5
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_str)

    aruco_tpl = cv2.aruco.DetectorParameters()
    aruco_params = cv2.aruco.DetectorParameters()
    # parameters used before finding contours
    aruco_params.adaptiveThreshConstant = 7  # def: 7
    aruco_params.adaptiveThreshWinSizeMin = 3  # def: 3
    aruco_params.adaptiveThreshWinSizeMax = 80  # def: 23
    aruco_params.adaptiveThreshWinSizeStep = 5  # def: 10
    # Reject quads where pairs of edges have angles that are close to straight or close to 180 degrees.
    # Zero means that no quads are rejected. (In radians)
    aruco_params.aprilTagCriticalRad = 0  # def: 0
    aruco_params.aprilTagDeglitch = 0  # def: 0, Only useful for very noisy images
    # When fitting lines to the contours, what is the maximum mean squared error allowed?
    # This is useful in rejecting contours that are far from being quad shaped;
    # rejecting these quads "early" saves expensive decoding processing.
    aruco_params.aprilTagMaxLineFitMse = 10  # def: 10 (def*PI/180)
    # how many corner candidates to consider when segmenting a group of pixels into a quad.
    aruco_params.aprilTagMaxNmaxima = 10  # def: 10
    # reject quads containing too few pixels
    aruco_params.aprilTagMinClusterPixels = 4  # def: 5
    # When we build our model of black & white pixels, we add an extra check that the white model must be
    # (overall) brighter than the black model. How much brighter? (in pixel values, [0,255])
    aruco_params.aprilTagMinWhiteBlackDiff = 5  # def: 5
    # Detection of quads can be done on a lower-resolution image, improving speed at a cost of
    # pose accuracy and a slight decrease in detection rate.
    # Decoding the binary payload is still done at full resolution.
    aruco_params.aprilTagQuadDecimate = 0.0  # def: 0.0
    # What Gaussian blur should be applied to the segmented image (used for quad detection?)
    # Parameter is the standard deviation in pixels. Very noisy images benefit from non-zero values (e.g. 0.8)
    aruco_params.aprilTagQuadSigma = 0.0  # def: 0.0
    # maximum number of iterations for stop criteria of the corner refinement process
    aruco_params.cornerRefinementMaxIterations = 30  # def: 30
    # corner refinement method.
    #   cv2.aruco.CORNER_REFINE_NONE: no refinement (default)
    #   cv2.aruco.CORNER_REFINE_SUBPIX: do subpixel refinement
    #   cv2.aruco.CORNER_REFINE_CONTOUR: use contour-Points
    #   cv2.aruco.CORNER_REFINE_APRILTAG: use the AprilTag2 approach
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    aruco_params.cornerRefinementMinAccuracy = 0.1  # def: 0.1
    aruco_params.cornerRefinementWinSize = 5  # def: 5 (in pixels)
    # error correction rate respect to the maximum error correction capability for each dictionary.
    aruco_params.errorCorrectionRate = 0.6  # def: 0.6
    # number of bits of the marker border, i.e. marker border width
    aruco_params.markerBorderBits = 1  # def: 1
    # number of allowed white bits in the border represented as a rate respect to the total number of bits per marker
    aruco_params.maxErroneousBitsInBorderRate = 0.7  # def: 0.35
    # minimum distance between corners for detected markers relative to its perimeter
    aruco_params.minCornerDistanceRate = 0.05  # def: 0.05
    aruco_params.minDistanceToBorder = 3  # def: 3 (in pixels)
    # minimum mean distance between two marker corners to be considered similar, so that the smaller one is removed.
    # The rate is relative to the smaller perimeter of the two markers
    aruco_params.minMarkerDistanceRate = 0.05  # def: 0.05
    # determine minimum perimeter for marker contour to be detected.
    # This is defined as a rate respect to the maximum dimension of the input image
    aruco_params.minMarkerPerimeterRate = 0.03  # def: 0.03
    # determine maximum perimeter for marker contour to be detected.
    # This is defined as a rate respect to the maximum dimension of the input image
    aruco_params.maxMarkerPerimeterRate = 4.0  # def: 4.0
    # minimum standard deviation in pixels values during the decodification step to apply
    # Otsu thresholding (otherwise, all the bits are set to 0 or 1 depending on mean higher than 128 or not)
    aruco_params.minOtsuStdDev = 5.0  # def: 5.0
    # width of the margin of pixels on each cell not considered for the determination of the cell bit.
    # Represents the rate respect to the total size of the cell, i.e. perspectiveRemovePixelPerCell
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13  # def: 0.13
    # number of bits (per dimension) for each cell of the marker when removing the perspective
    aruco_params.perspectiveRemovePixelPerCell = 1  # def: 4
    # minimum accuracy during the polygonal approximation process to determine which contours are squares.
    aruco_params.polygonalApproxAccuracyRate = 0.03  # def: 0.03

    img_tpl = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    detector_tpl = cv2.aruco.ArucoDetector(aruco_dict, aruco_tpl)
    (corners_tpl, ids_tpl, rejected_tpl) = detector_tpl.detectMarkers(img_tpl)
    dict_template = {}
    for corner, [id] in zip(corners_tpl, ids_tpl):
        dict_template[id] = corner

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    marker_images, reply = get_marker_images(inpath,
                                             aruco_detector=detector,
                                             filename_pattern=filename_pattern,
                                             verbose=verbose,
                                             debug=debug)
    if reply['status'] != MESSAGE_STATUS['SUCCESS']:
        return reply

    # get the marker corners with the smallest transformation error
    error_opt = np.inf
    H_opt = np.identity(3)
    timings = []
    for f, params in marker_images.items():
        dst_pts = []
        src_pts = []
        src_id = []
        if f == template:
            continue
        count_corners = 0
        for id, corner in params.items():
            dst_pts.extend([p for a in dict_template[id] for b in a for p in b])
            src_pts.extend([p for a in corner for b in a for p in b])
            src_id.extend([id])
            count_corners += 1
        if count_corners < 4:
            print_msg(f"Warning!  found only {count_corners} markers", "YELLOW")

        dst_pts = np.array(dst_pts, dtype=np.float32).reshape((-1, 1, 2))
        src_pts = np.array(src_pts, dtype=np.float32).reshape((-1, 1, 2))
        src_id = np.array(src_id)
        H, mask = cv2.findHomography(
            srcPoints=src_pts,
            dstPoints=dst_pts,
            method=cv2.LMEDS,  # [cv2.LMEDS (def), cv2.RANSAC, cv2.RHO]
            ransacReprojThreshold=1.0,  # def: 3
            maxIters=4000,  # def: 2000
            confidence=0.9995)  # def: 0.995

        # if debug:
        #     img_raw = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        #     img_timer = cv2.warpPerspective(img_raw, H, (img_tpl.shape[1], img_tpl.shape[0]), flags=cv2.INTER_LINEAR)
        #     img_timer_3C = cv2.cvtColor(img_timer, cv2.COLOR_GRAY2RGB)
        #     src_id = np.array([[i] for i in range(3, -1, -1)])
        #     src_markers = np.split(np.array(src_pts, dtype=np.float32).ravel().reshape((-1, 4, 2)), count_corners)
        #     cv2.aruco.drawDetectedMarkers(img_raw, src_markers)
        #     cv2.imshow("src", img_raw)

        # convert the src_pts with H and display them on the img_timer
        # if debug:
        #     dst_markers = np.split(np.array(dst_pts, dtype=np.float32).ravel().reshape((-1, 4, 2)), count_corners)
        #     cv2.aruco.drawDetectedMarkers(img_timer_3C, dst_markers, src_id, borderColor=(255, 0, 0))
        timer_src = cv2.perspectiveTransform(src=src_pts, m=H)
        error = np.sum(abs(timer_src - dst_pts))
        if error < error_opt:
            error_opt = error
            H_opt = H

        # if debug:
        #     src_markers = np.split(np.array(timer_src, dtype=np.float32).ravel().reshape((-1, 4, 2)), count_corners)
        #     cv2.aruco.drawDetectedMarkers(img_timer_3C, src_markers, src_id)
        #     cv2.imshow("dst", img_timer_3C)
        #     user_input = cv2.waitKey()
        #     if user_input == ord('\x1b'):
        #         break  # '\x1b': escape key
    print_msg(f"smallest error (summed absolute distance from template corner coordinates): {error_opt}",
                "DARK_GREEN")

    print_msg(f"Get timings from images in {inpath}...", "BLUE")
    for file in marker_images.keys():
        seconds, hertz, user_input, reply = get_time(file,
                                                     H_opt, (img_tpl.shape[1], img_tpl.shape[0]),
                                                     debug=debug,
                                                     verbose=verbose)
        if reply['status'] != MESSAGE_STATUS['SUCCESS']:
            return reply
        timings.append([os.path.basename(file), seconds, hertz, 0])
        if user_input == ord('\x1b'):
            break  # '\x1b': escape key

    write_timings_to_csv(inpath, "timings_raw.csv", timings)
    # do some error checking on the timing result using adjacent time result as a reference
    timings, reply = fix_time_errors(timings)
    if reply['status'] == MESSAGE_STATUS['SUCCESS']:
        write_timings_to_csv(inpath, "timings.csv", timings)
    else:
        print_msg(f"Warning!  'timings.csv not written to: {inpath}", "YELLOW")

    return return_value(MESSAGE_STATUS['SUCCESS'], message="mocap_timing_detector ran successfully")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="generates timings.csv in 'inpath' from detecting the timing app in images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inpath", type=str, required=True, help="path to folder with timing images")
    parser.add_argument("--pattern", type=str, default="*.png", help="a 'glob' consumable filename pattern")
    parser.add_argument("--template", type=str, required=True, help="filepath to template image")
    parser.add_argument("--max_hz", type=int, default=MONITOR_MAX_HZ, help="maximum frequency of the monitor")
    parser.add_argument("--use_existing_raw",
                        type=make_boolean,
                        default=True,
                        help="process timings_raw.csv if it already exists")
    parser.add_argument("-v",
                        "--verbose",
                        type=make_boolean,
                        default=False,
                        help="print additional information to console")
    parser.add_argument("-d",
                        "--debug",
                        type=make_boolean,
                        default=False,
                        help="show images with marker/cell information")
    args = parser.parse_args()

    reply = mocap_timing_detector(inpath=args.inpath,
                                  template=args.template,
                                  filename_pattern=args.pattern,
                                  max_hz=args.max_hz,
                                  use_existing_raw=args.use_existing_raw,
                                  debug=args.debug,
                                  verbose=args.verbose)

    print_msg(reply, "GREEN" if reply['status'] == MESSAGE_STATUS["SUCCESS"] else "RED")


if __name__ == "__main__":
    main()
