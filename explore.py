import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import coherence
from scipy.signal import periodogram
from scipy.signal import welch

import support as sp


class StationExplorer:
    """ Класс для исследования станции.
    - station_name: имя станции
    - height: глубина станции
    - az_path: путь до Аz-файла 
    - p_path: путь до Р-файла 
    - p_start, p_stop: время измерений Р
    - az_start, az_stop: время измерений Az
    - period: рассматриваемый рабочий промежуток в часах
    - start_point: начало рабочего периода для станции
    """
    def __init__(self, station_name, height,
                 az_path, p_path, 
                 p_start, p_stop, 
                 az_start, az_stop,
                 period=1, start_point=None):
        self.station_name = station_name
        self.period = period
        self.start_point = start_point
        self.az_path, self.p_path = az_path, p_path
        self.p_start, self.p_stop = p_start, p_stop
        self.az_start, self.az_stop = az_start, az_stop
        
        self.freq_gravity = 0.366 * np.sqrt(9.80665/height)
        self.freq_acoustic = 1500 / 4 / height
        self.freq_experimental = 0.1
    
    def run(self):
        sns.set_theme()
        self.load_data()
        if not self.start_point:
            self.draw_basic()                                                               # построение начальных графиков
            self.start_point = int(input('Введите промежуток начала колебаний: '))          # выбор точки начала анализа
        self.az_osc = self.az[self.start_point: self.start_point+36000*self.period]         # выделение рабочих участков
        self.p_osc = self.p[self.start_point: self.start_point+36000*self.period]
        self.analyze()                                                                      # анализ станции
        
        
    def load_data(self) -> None:
        self.res_start = sp.min_time(self.p_start, self.az_start)                           # общее начальное время
        self.res_stop = sp.max_time(self.p_stop, self.az_stop)                              # общее конечное время
        print(f'Время начала измерений: {"-".join(map(str, self.res_start))}', 
              f'Время конца измерений: {"-".join(map(str, self.res_stop))}', sep='\n')

        self.az, self.az_mean = sp.read_file(self.az_path, float)                           # исследуемый массив ускорений
        self.p, self.p_mean = sp.read_file(self.p_path, lambda x: int(x) / 10)              # исследуемый массив давлений
        self.theoretical_p_ratio = (self.p_mean/9.80665)**2                                 # для графика отношения мощностей


        # 1 сек = 10 промежутков, 1 мин = 600 промежутков, 1 час = 36000 промежутков  
        dur_time = sp.time_delta(self.res_stop, self.res_start)                             # Длительность всех измерений
        dur_len = dur_time[0] * 36000 + dur_time[1] * 600                                   # Длительность в промежутках

        # Дополняем меньший из промежутков
        if self.res_stop == self.p_stop:
            delta_len = sp.time_delta(self.res_stop, self.az_stop)                          # Разница 
            self.az += [0 for x in range(delta_len[0] * 36000 + delta_len[1] * 600)]        # Заполняем соотв часть нулями
        else:
            delta_len = sp.time_delta(self.res_stop, self.p_stop) 
            self.p += [0 for x in range(delta_len[0] * 36000 + delta_len[1] * 600)] 

        if self.res_start == self.p_start:
            delta_len = sp.time_delta(self.az_start, self.res_start) 
            self.az = [0 for x in range(delta_len[0] * 36000 + delta_len[1] * 600)] + self.az 
        else:
            delta_len = sp.time_delta(self.p_start, self.res_start)
            self.p = [0 for x in range(delta_len[0] * 36000 + delta_len[1] * 600)] + p

        print(f'Проверка на равенство дополненных промежутков: {len(self.az) == len(self.p)}')

        
    def draw_basic(self):
        """ Построение начальных графиков расширенных и нормализованных данных.
        """
        az_dic = {
            'title': f'Норм расширенные данные Az - {self.station_name}',
            'xlabel': 'Время, 1c/10',
            'ylabel': 'Вертикальное ускорение Az'
        }
        p_dic = {
            'title': f'Норм расширенные данные P - {self.station_name}',
            'xlabel': 'Время, 1c/10',
            'ylabel': 'Давление столбца жидкости на 1м2'
        }
        
        fig, axs = plt.subplots(ncols=2, figsize=(15,4))
        sns.lineplot(data=self.az, ax=axs[0]).set(**az_dic)
        sns.lineplot(data=self.p, ax=axs[1]).set(**p_dic)
        plt.show()

        
    def analyze(self):
        """ Функция анализа станции и построения MSC и S_Ratio графиков, 
            а также нахождения доли хороших частот и дельта 
        """
       
        self.freq_array, self.MSC_array = coherence(self.az_osc, self.p_osc, fs=10, nperseg=8192)
        self.freq1, power_spectral_density_az = welch(self.az_osc, fs=10, nperseg=8192)
        self.freq2, power_spectral_density_p = welch(self.p_osc, fs=10, nperseg=8192)
        self.relation_Sp_and_Saz = power_spectral_density_p / power_spectral_density_az
        
        #Доля частот и ДЕЛЬТА
        df = pd.DataFrame(data=[self.freq_array, self.MSC_array, self.relation_Sp_and_Saz]).T.rename(columns={0: 'freq', 1: 'msc', 2: 'ratio'})
        df_results = df.loc[(df.freq > self.freq_gravity) & (df.freq < self.freq_experimental)]
        df_good_freq = df_results.loc[df_results.msc >= 0.99]
        average_ratio = df_good_freq.ratio.mean()
        delta = np.sqrt(average_ratio) / np.sqrt(self.theoretical_p_ratio) - 1

        print(f'Доля хороших частот = {round(df_good_freq.shape[0]/df_results.shape[0], 3)}')
        print(f'Дельта = {round(delta, 3)}')
        
        az_dic = {
            'title': f'Рабочий участок Az - {self.station_name}',
            'xlabel': 'Время, 1c/10',
            'ylabel': 'Вертикальное ускорение Az'
        }
        p_dic = {
            'title': f'Рабочий участок P - {self.station_name}',
            'xlabel': 'Время, 1c/10',
            'ylabel': 'Давление столбца жидкости на 1м2'
        }
        msc_dic = {
            'title': f'MSC - {self.station_name}',
            'xlabel': 'частота, Гц',
            'ylabel': 'MSC',
            'xscale': 'log'
        }
        psd_dic = {
            'title': f'Отношение спектров мощностей - {self.station_name}',
            'xlabel': 'частота, Гц',
            'ylabel': 'S_p / S_Az',
            'xscale': 'log',
            'yscale': 'log'
        }
        
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
        sns.lineplot(data=self.az_osc, ax=axs[0][0]).set(**az_dic)
        sns.lineplot(data=self.p_osc, ax=axs[0][1]).set(**p_dic)
        
        # MSC
        sns.lineplot(x=self.freq_array, y=self.MSC_array, ax=axs[1][0]).set(**msc_dic)
        axs[1][0].axvline(x=self.freq_acoustic, c='g', linestyle=':')
        axs[1][0].axvline(x=self.freq_gravity, c='g', linestyle=':')
        axs[1][0].axvline(x=self.freq_experimental, c='g', linestyle=':')
        axs[1][0].set_xlim([1e-2, 25e-2])
        
        
        # Power spectral density ratio
        sns.lineplot(x=self.freq1, y=self.relation_Sp_and_Saz, ax=axs[1][1]).set(**psd_dic)
        axs[1][1].set_xlim([1e-2, 25e-2])
        axs[1][1].set_ylim([1e12, 1e14])
        axs[1][1].axvline(x=self.freq_acoustic, c='g', linestyle=':')
        axs[1][1].axvline(x=self.freq_gravity, c='g', linestyle=':')
        axs[1][1].axvline(x=self.freq_experimental, c='g', linestyle=':')
        axs[1][1].axhline(y=self.theoretical_p_ratio, c='g', linestyle=':')
        
        
        plt.show()
        
        

        # data_to_append = [[station_name, 
        #                   df_good_freq.shape[0]/df_results.shape[0],
        #                   delta]]
        # with open('final_stations_data.csv', 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(data_to_append)