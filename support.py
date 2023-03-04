def time_delta(time1, time2):
    '''Вычисляет time1 - time2'''
    duration_in_min = time1[0] * 60 + time1[1] - time2[0] * 60 - time2[1]
    return duration_in_min // 60, duration_in_min % 60