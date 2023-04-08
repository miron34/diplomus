import numpy as np
import math


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
    # точно ли надо нормализовать??? для получения ускорения a_total
    # либо сначала взять три массива потом посчитать суммарный а потом нормализовать?
    
    
def read_file2(path: str, transform) -> (list, float):
    """ читаем файл, обрабатываем его функцией transform
    """
    f = open(path, "r")
    seq = [transform(x) for x in f.read().split()] # численные значения
    return seq

def find_distance_in_meters(llat1,llong1,llat2,llong2):
        """ Функция нахождения расстояния в метрах между двумя
            точками на сфере.
        """
        # радиус сферы (Земли)
        rad = 6372795

        # в радианах
        lat1, lat2 = llat1*math.pi/180., llat2*math.pi/180.
        long1, long2 = llong1*math.pi/180., llong2*math.pi/180.

        # косинусы и синусы широт и разницы долгот
        cl1, cl2 = math.cos(lat1), math.cos(lat2)
        sl1, sl2 = math.sin(lat1), math.sin(lat2)
        delta = long2 - long1
        cdelta = math.cos(delta)
        sdelta = math.sin(delta)

        # вычисления длины большого круга
        y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
        x = sl1*sl2+cl1*cl2*cdelta
        ad = math.atan2(y,x)
        dist = ad*rad

        #вычисление начального азимута
        x = (cl1*sl2) - (sl1*cl2*cdelta)
        y = sdelta*cl2
        z = math.degrees(math.atan(-y/x))

        if (x < 0):
            z = z+180.

        z2 = (z+180.) % 360. - 180.
        z2 = - math.radians(z2)
        anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi))) )
        angledeg = (anglerad2*180.)/math.pi

        return dist

