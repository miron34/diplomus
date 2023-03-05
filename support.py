import numpy as np


def time_delta(time1: tuple, time2: tuple) -> tuple:
    """ Вычисляет time1 - time2
    """
    duration_in_min = time1[0] * 60 + time1[1] - time2[0] * 60 - time2[1]
    return duration_in_min // 60, duration_in_min % 60

def min_time(time1: tuple, time2: tuple) -> tuple:
    if time1[0] < time2[0]: return time1
    elif time1[0] > time2[0]: return time2
    elif time1[1] <= time2[1]: return time1
    else: return time2

def max_time(time1: tuple, time2: tuple) -> tuple:
    if time1[0] > time2[0]: return time1
    elif time1[0] < time2[0]: return time2
    elif time1[1] >= time2[1]: return time1
    else: return time2

def read_file(path: str, transform) -> (list, float):
    """ читаем файл, обрабатываем его функцией transform
    """
    f = open(path, "r")
    seq = [transform(x) for x in f.read().split()] # численные значения
    mean = np.mean(seq)                            # считаем среднее значение ускорения на всем промежутке
    seq = [x - mean for x in seq]                  # нормализация
    return seq, mean

