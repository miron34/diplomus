from support import *

from tqdm import tqdm
import numpy as np
import pandas as pd

class Relief:
    
    @staticmethod
    def find_closest_grad(x, y, df):
        '''
        x - current longitude
        y - current latitude
        returns: grad of nearest point
        '''
        closest = None
        closest_point = []

        for idx in df.index:
            x1 = df.loc[idx, 'longitude']
            y1 = df.loc[idx, 'latitude']
            z1 = df.loc[idx, 'depth']
            grad = df.loc[idx, 'grad']
            grad_east = df.loc[idx, 'grad_east']
            grad_north = df.loc[idx, 'grad_north']
            # metric = (x-x1)**2 + (y-y1)**2
            metric = find_distance_in_meters(y,x,y1,x1)
            if (closest == None) or (closest > metric):
                closest = metric
                closest_point = [x1, y1, z1, grad, grad_east, grad_north]

        return closest_point
    
    
    # граница для "крутого склона" взята на глаз = 0.1
    @staticmethod
    def find_closest_slope(x, y, df, grad_threshold=0.1):
        '''
        x - current longitude
        y - current latitude
        returns: distance to nearest slope 
        '''
        closest = None
        closest_point = []

        df = df[df.grad > grad_threshold]

        for idx in df.index:
            x1 = df.loc[idx, 'longitude']
            y1 = df.loc[idx, 'latitude']
    #         metric = (x-x1)**2 + (y-y1)**2
            metric = find_distance_in_meters(y,x,y1,x1)
            if (closest == None) or (closest > metric):
                closest = metric
                closest_point = [x1, y1]

        if pd.isna(closest):
            return None
        else:
            return round(closest)
        
    
    def __init__(self, relief_path, depth_stations_path):
        
        
        # загрузка файла рельефа местности
        sed = np.loadtxt(relief_path, unpack = True)
        self.map_data = pd.DataFrame(sed).T.rename(columns={
            0:'longitude',
            1:'latitude',
            2:'depth'
        })

        # загрузка файла глубин станций
        stations_all = pd.read_csv(depth_stations_path, delimiter=';', header=None).rename(columns={
            0:'longitude',
            1:'latitude',
            2:'depth',
            3:'station'
        })
        stations_all['depth'] = stations_all['depth'] * (-1)
        self.stations = stations_all[(stations_all.longitude >= self.map_data.longitude.min()) & 
                                (stations_all.longitude <= self.map_data.longitude.max()) & 
                                (stations_all.latitude >= self.map_data.latitude.min()) &
                                (stations_all.latitude <= self.map_data.latitude.max()+1)].copy()

        self.stations['station'] = self.stations['station'].apply(lambda s: s[2:])
        self.stations = self.stations.loc[:17,].copy()    
        
        
        
        # создаем словарь глубин для более быстрого поиска глубины по коор-там
        self.depth_dict = {}
        for idx in self.map_data.index:
            self.depth_dict[str([self.map_data.loc[idx, 'longitude'],
                       self.map_data.loc[idx, 'latitude']])] = self.map_data.loc[idx, 'depth']

        # оси Х-У сетки
        x_axe = sorted(self.map_data.longitude.unique())
        y_axe = sorted(self.map_data.latitude.unique())
        
        
        
        
        
        # расчет градиентов для каждой точки поверхности UPDATED
        shift = 1
        self.map_data['grad'] = None
        self.map_data['grad_east'] = None
        self.map_data['grad_north'] = None

        for idx in tqdm(self.map_data.index):

            current_X = self.map_data.loc[idx, 'longitude']
            current_Y = self.map_data.loc[idx, 'latitude']
            current_depth = self.map_data.loc[idx, 'depth']

            try:
                # точка справа по коорд сетке
                A2_x = x_axe[x_axe.index(current_X) + shift]
                A2_y = current_Y
                A2_depth = self.depth_dict[str([A2_x, A2_y])]

                # точка сверху по коорд сетке
                B2_x = current_X
                B2_y = y_axe[y_axe.index(current_Y) + shift]
                B2_depth = self.depth_dict[str([B2_x, B2_y])]

                # точка слева по коорд сетке
                A1_x = x_axe[x_axe.index(current_X) - shift]
                A1_y = current_Y
                A1_depth = self.depth_dict[str([A1_x, A1_y])]

                # точка снизу по коорд сетке
                B1_x = current_X
                B1_y = y_axe[y_axe.index(current_Y) - shift]
                B1_depth = self.depth_dict[str([B1_x, B1_y])]

                # Х-составлящая градиента (East-gradient)
                grad_X = ((A2_depth - A1_depth) / 
                          find_distance_in_meters(A2_y, A2_x, A1_y, A1_x))

                # Y-составляющая градиента (North-gradient)
                grad_Y = ((B2_depth - B1_depth) / 
                          find_distance_in_meters(B2_y, B2_x, B1_y, B1_x))
                
                # суммарный градиент
                gradient = np.sqrt(grad_X**2 + grad_Y**2)
                self.map_data.at[idx, 'grad'] = gradient
                self.map_data.at[idx, 'grad_east'] = grad_X
                self.map_data.at[idx, 'grad_north'] = grad_Y

            # для обхода крайних точек без нужного числа соседей
            except IndexError:
                continue
                
                
                
        # находим ближайший рассчитанный градиент к каждой станции        
        self.stations['grad'] = self.stations.apply(lambda row: self.find_closest_grad(row.longitude, 
                                                                row.latitude, 
                                                                df=self.map_data)[3], axis=1)
        self.stations['grad_east'] = self.stations.apply(lambda row: self.find_closest_grad(row.longitude, 
                                                                row.latitude, 
                                                                df=self.map_data)[4], axis=1)
        self.stations['grad_north'] = self.stations.apply(lambda row: self.find_closest_grad(row.longitude, 
                                                                row.latitude, 
                                                                df=self.map_data)[5], axis=1)
        
        # находим расстояние до ближайшего крутого склона
        self.stations['from_slope'] = self.stations.apply(lambda row: self.find_closest_slope(row.longitude, 
                                              row.latitude, 
                                              df=self.map_data), axis=1)
        
        # находим расстояние в глубинах станций
        self.stations['H_depth'] = self.stations['from_slope'] // abs(self.stations['depth'])